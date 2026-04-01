"""Optimized haplotype survey pipeline.

This module implements the core stages of the survey pipeline, including
haplotype block construction, background variant integration, batch sequence
extraction, and parallel delta scoring.
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from copy import deepcopy
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from pandera.typing import DataFrame
from tqdm import tqdm

from panthera.core.bio.blocks import HaplotypeBlock, VariantSchema
from panthera.core.bio.extend_phaseset import extend_phaseset
from panthera.core.bio.gene import find_genes_at_pos
from panthera.core.bio.parse_bg_vcf import VCFCoordinates
from panthera.core.bio.parse_genome import GenomeParser
from panthera.core.bio.split_by_haplotype import split_by_haplotype
from panthera.core.bio.wig import generate_wig

from panthera.core.ssp.calc_delta import SSPScorer


from panthera.utils.exceptions import (
    DataResolutionError,
    BackgroundConflictError,
    AmbiguousDeletionError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

_DELTA_CHUNKSIZE: int = 32  # ProcessPoolExecutor.map chunksize for delta scoring
_DEFAULT_GPU_BATCH: int = 16  # sequences per GPU call (wt + mt interleaved)
_DEFAULT_IO_THREADS: int = 8  # threads for background-VCF fetching


# ---------------------------------------------------------------------------
# Inter-phase data containers
# ---------------------------------------------------------------------------


@dataclass
class _BlockSeqs:
    """Links a HaplotypeBlock to its extracted and pre-processed sequences.

    Attributes:
        block: Metadata for the genomic region.
        wt_seq: Raw wild-type sequence with INDEL markers.
        mt_seq: Raw mutant sequence with INDEL markers.
        wt_seq_clean: Cleaned WT sequence for model submission.
        mt_seq_clean: Cleaned MT sequence for model submission.
        reverse_output: Whether the block is on the minus strand.
        extraction_start: 1-based genomic start position.
    """

    block: HaplotypeBlock  # HaplotypeBlock — kept for metadata only
    wt_seq: str  # raw WT  (with indel markers, used for delta scoring)
    mt_seq: str  # raw MT  (with indel markers, used for delta scoring)
    wt_seq_clean: str  # model input — indel markers stripped, rc applied if (−)
    mt_seq_clean: str  # model input — indel markers stripped, rc applied if (−)
    reverse_output: bool  # True when block is on the (−) strand
    extraction_start: int  # 1-based start on the chromosome (start_bound)


@dataclass
class _BlockPredictions:
    """Minimal picklable struct consumed by the delta-scoring worker.

    Attributes:
        chrom_start: Genomic start position.
        splice_sites: Known splice sites in the region.
        wt_seq: WT sequence with markers.
        mt_seq: MT sequence with markers.
        wt_acc: Array of WT acceptor probabilities.
        wt_dnr: Array of WT donor probabilities.
        mt_acc: Array of MT acceptor probabilities.
        mt_dnr: Array of MT donor probabilities.
        extraction_start: Coordinate mapping start position.
        block_type: Identifier for the block type.
        chrom: Chromosome name.
        end: Genomic end position.
        strand: Gene strand.
        gene_name: Target gene name.
        gene_id: Target gene ID.
        population: Genetic background population.
        background_id: Sample identifier.
        haplotype_id: Haplotype identifier (A/B).
        block_id: Unique block ID.
        block_name: Human-readable block name.
    """

    # ---- SSPScorer inputs ----
    chrom_start: int
    splice_sites: dict[str, list[int]]
    wt_seq: str
    mt_seq: str
    wt_acc: np.ndarray
    wt_dnr: np.ndarray
    mt_acc: np.ndarray
    mt_dnr: np.ndarray
    extraction_start: int
    block_type: str

    # ---- Result-row metadata ----
    chrom: str
    end: int
    strand: str
    gene_name: str
    gene_id: str
    population: str
    background_id: str
    haplotype_id: str
    block_id: str
    block_name: str


# ---------------------------------------------------------------------------
# Module-level worker (must be importable by child processes)
# ---------------------------------------------------------------------------


def _compute_delta_scores(pred: _BlockPredictions) -> dict:
    """Calculates delta scores for a block (worker function).

    Args:
        pred: Prediction data structure.

    Returns:
        dict: A dictionary of results formatted as a TSV row.
    """
    delta_scorer = SSPScorer(
        chrom_start=pred.chrom_start,
        splice_sites=pred.splice_sites,
        wt_seq=pred.wt_seq,
        mt_seq=pred.mt_seq,
        wt_acc=pred.wt_acc,
        wt_dnr=pred.wt_dnr,
        mt_acc=pred.mt_acc,
        mt_dnr=pred.mt_dnr,
    )
    delta_scorer.align_prob()

    logger.debug(f"PREDICTION: WT SEQ LENGTH {len(pred.wt_seq)}")
    logger.debug(f"PREDICTION: MT SEQ LENGTH {len(pred.mt_seq)}")

    raw_deltas = delta_scorer.calc_raw_deltas()
    masked_deltas = delta_scorer.calc_masked_deltas()

    max_raw = round(float(np.max(raw_deltas)), 3)
    max_masked = round(float(np.max(masked_deltas)), 3)

    return {
        "chrom": pred.chrom,
        "start": pred.chrom_start,
        "end": pred.end,
        "strand": pred.strand,
        "gene_name": pred.gene_name,
        "gene_id": pred.gene_id,
        "population": pred.population,
        "genetic_background": pred.background_id,
        "haplotype_index": pred.haplotype_id,
        "block_ID": pred.block_id,
        "block_type": pred.block_type,
        "block_variants": pred.block_name,
        "raw_delta_pos": delta_scorer._find_max_delta_locations(raw_deltas, max_raw),
        "masked_delta_pos": delta_scorer._find_max_delta_locations(
            masked_deltas, max_masked
        ),
        "raw_delta": max_raw,
        "masked_delta": max_masked,
    }


def _generate_wig(outdir: str, pred: _BlockPredictions) -> None:
    try:
        generate_wig(
            gene_name=pred.gene_name,
            background_id=pred.background_id,
            haplotype_id=pred.haplotype_id,
            chrom=pred.chrom,
            start=pred.extraction_start,
            outdir=outdir,
            wt_acc=pred.wt_acc,
            wt_dnr=pred.wt_dnr,
            mt_acc=pred.mt_acc,
            mt_dnr=pred.mt_dnr,
            block_id=pred.block_id,
            block_type=pred.block_type,
        )
    except ValueError as e:
        logger.error(f"Error generating WIG file for block {pred.block_id}: {e}")


# ---------------------------------------------------------------------------
# Pipeline class  (replace the existing method body with these)
# ---------------------------------------------------------------------------


def phase1_build_blocks(
    contiguous_vdfs: list,
    gtf_dict: dict,
    block_extension: int,
) -> tuple[list, list]:
    """Phase 1: Build haplotype and single-variant blocks from VCF data.

    Args:
        contiguous_vdfs: List of DataFrames, each containing a contiguous
            set of variants.
        gtf_dict: Parsed GTF metadata for gene lookups.
        block_extension: Distance (bp) to extend phase sets for homozygous
            variants.

    Returns:
        tuple[list, list]: A tuple containing (haplotype_blocks,
            single_variant_blocks).
    """
    haplotype_blocks: list = []
    single_variant_blocks: list = []

    for c_vdf in contiguous_vdfs:
        unique_pairs = c_vdf[["chrom", "phase_set"]].drop_duplicates()
        pbar = tqdm(
            total=len(unique_pairs),
            desc="Phase 1 — building haplotype blocks",
            leave=True,
        )
        for chrom, ps in unique_pairs.itertuples(index=False):
            current_vdf = cast(
                DataFrame[VariantSchema],
                extend_phaseset(
                    c_vdf,
                    chrom=chrom,
                    ps_id=ps,
                    ext_len=block_extension,
                ),
            )

            # Collect gene objects for every position in the phase set
            gene_objs: list = []
            for pos in current_vdf["pos"].unique():
                gene_objs += find_genes_at_pos(
                    chrom=chrom,
                    pos=pos,
                    gtf_dict=gtf_dict,
                    existing_genes=gene_objs,
                )

            for gene_obj in gene_objs:
                # One single-variant block per row in the phase set
                for i, variant_df in enumerate(
                    np.array_split(current_vdf, len(current_vdf))
                ):
                    svb = HaplotypeBlock(
                        variants_df=cast(DataFrame[VariantSchema], variant_df),
                        gene_obj=gene_obj,
                    )
                    svb.population = "BASE"
                    svb.background_id = "BASE"
                    svb.haplotype_id = "NA"
                    svb.block_type = "SINGLE_VARIANT"
                    # Use a descriptive ID: Gene_Pos
                    # Explicitly cast variant_df to avoid NDArray inference from array_split
                    vdf_slice = cast(pd.DataFrame, variant_df)
                    svb.block_id = f"S{vdf_slice['pos'].iloc[0]}"
                    single_variant_blocks.append(svb)

                # One full-phase-set block per gene
                hb = HaplotypeBlock(variants_df=current_vdf, gene_obj=gene_obj)
                hb.population = "BASE"
                hb.background_id = "BASE"
                hb.haplotype_id = "NA"
                hb.block_type = "HAPLOTYPE"
                hb.block_id = "H0"
                haplotype_blocks.append(hb)

            current_vdf = None
            pbar.update()

    return haplotype_blocks, single_variant_blocks


# ------------------------------------------------------------------
# Phase 2 — Add genetic background  (ThreadPoolExecutor for I/O)
# ------------------------------------------------------------------


def _fetch_one_background(
    block,
    gbs: str,
    gb_group_name: str,
    bg_vcf_manager,
    resolve_conflicts: bool,
) -> list:
    """Worker function to fetch background variants for a single block and sample.

    Args:
        block: The target HaplotypeBlock.
        gbs: Sample identifier.
        gb_group_name: Population group name.
        bg_vcf_manager: Manager for fetching from background VCFs.
        resolve_conflicts: Whether to resolve overlapping variants.

    Returns:
        list: A list of new HaplotypeBlocks with background variants added.
    """
    coords = VCFCoordinates(
        chrom=block.chrom,
        start=block.max_start,
        end=block.min_end,
    )

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        try:
            bg_vdf = bg_vcf_manager.fetch_region(sample_id=gbs, coords=coords)
        except DataResolutionError as exc:
            logger.warning(
                "Failed to read background VCF for %s: %s. Skipping.", gbs, exc
            )
            raise  # re-raise; caller ignores DataResolutionError futures

    if warning_list:
        logger.warning("No variants found in background VCF for %s. Skipping.", gbs)
        return []

    contiguous_bg_vdfs = split_by_haplotype(cast(DataFrame[VariantSchema], bg_vdf))

    result: list = []
    for hap_id, c_bg_vdf in zip(("A", "B"), contiguous_bg_vdfs):
        target_bg_block = deepcopy(block)
        try:
            target_bg_block.add_background_variants(
                background_df=c_bg_vdf,
                population=gb_group_name,
                background_id=gbs,
                haplotype_id=hap_id,
                resolve_conflicts=resolve_conflicts,
            )
            # Ensure unique block_id for background blocks
            parent_id = getattr(block, "block_id", "UNK")
            target_bg_block.block_id = parent_id
            result.append(target_bg_block)
        except BackgroundConflictError as exc:
            logger.warning(
                "Variant conflict in background VCF for %s: %s. Skipping.", gbs, exc
            )
        except AmbiguousDeletionError as exc:
            logger.warning(
                "Ambiguous deletion in background VCF for %s: %s. Skipping.", gbs, exc
            )
    return result


def phase2_add_background(
    haplotype_blocks: list,
    gb_samples: tuple,
    gb_group_name: str,
    bg_vcf_manager,
    resolve_conflicts: bool,
    n_threads: int = _DEFAULT_IO_THREADS,
) -> list:
    """Phase 2: Incorporate genetic background variants in parallel.

    Args:
        haplotype_blocks: List of primary haplotype blocks.
        gb_samples: Tuple of sample IDs to fetch.
        gb_group_name: Population group name.
        bg_vcf_manager: Background VCF manager instance.
        resolve_conflicts: Whether to resolve variant overlaps.
        n_threads: Number of I/O threads to use.

    Returns:
        list: Consolidated list of blocks including background variants.
    """
    work_items = [(block, gbs) for block in haplotype_blocks for gbs in gb_samples]
    target_background_blocks: list = []

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        future_map = {
            executor.submit(
                _fetch_one_background,
                block,
                gbs,
                gb_group_name,
                bg_vcf_manager,
                resolve_conflicts,
            ): (block, gbs)
            for block, gbs in work_items
        }
        with tqdm(
            total=len(future_map),
            desc="Phase 2 — adding genetic background (parallel I/O)",
            leave=True,
        ) as pbar:
            for future in as_completed(future_map):
                try:
                    target_background_blocks.extend(future.result())
                except DataResolutionError:
                    pass  # already logged in the worker
                pbar.update()

    return target_background_blocks


# ------------------------------------------------------------------
# Phase 3 — Extract WT / MT sequences  (chrom-sorted for minimal I/O)
# ------------------------------------------------------------------


def phase3_extract_sequences(
    all_blocks: list,
    ssp_manager,
    genome_path: str,
    context_dist: int,
) -> list[_BlockSeqs]:
    """Phase 3: Extract WT and MT sequences for all blocks.

    Processes blocks in chromosome order to optimize FASTA access.

    Args:
        all_blocks: List of all HaplotypeBlocks to process.
        ssp_manager: Model manager for sequence preprocessing.
        genome_path: Path to the reference genome FASTA.
        context_dist: Total context distance for extraction.

    Returns:
        list[_BlockSeqs]: List of extracted sequence containers.
    """
    genome_parser = GenomeParser()

    # Sort guarantees a single FASTA load per chromosome
    all_blocks.sort(key=lambda b: b.chrom)

    block_seqs: list[_BlockSeqs] = []
    previous_chrom: str | None = None
    chrom_seq: str | None = None

    for block in tqdm(all_blocks, desc="Phase 3 — extracting sequences", leave=True):
        current_chrom = block.chrom
        if current_chrom != previous_chrom:
            chrom_seq = genome_parser.parse_genome(
                genome_path=genome_path, chrom=current_chrom
            )[current_chrom]
            previous_chrom = current_chrom

        try:
            # Re-calculate extraction start for coordinate mapping
            start_bound = max(1, int(block.vdf["pos"].min()) - context_dist // 2)

            wt_seq, mt_seq = block.extract_seqs(
                chrom_seq=chrom_seq,
                extension_len=context_dist // 2,
            )
        except AmbiguousDeletionError as exc:
            logger.warning(
                "Ambiguous deletion for block %s: %s. Skipping.", block.name, exc
            )
            continue

        wt_clean = ssp_manager.remove_indel_markers([wt_seq])[0]
        mt_clean = ssp_manager.remove_indel_markers([mt_seq])[0]

        if block.gene_obj.strand == "-":
            wt_clean = ssp_manager.reverse_complement([wt_clean])[0]
            mt_clean = ssp_manager.reverse_complement([mt_clean])[0]
            reverse_output = True
        else:
            reverse_output = False

        block_seqs.append(
            _BlockSeqs(
                block=block,
                wt_seq=wt_seq,
                mt_seq=mt_seq,
                wt_seq_clean=wt_clean,
                mt_seq_clean=mt_clean,
                reverse_output=reverse_output,
                extraction_start=start_bound,
            )
        )

    return block_seqs


# ------------------------------------------------------------------
# Phase 4 — Batch GPU prediction  (the largest single speedup)
# ------------------------------------------------------------------


def phase4_batch_predict(
    block_seqs: list[_BlockSeqs],
    ssp_manager,
    gpu_batch_size: int = _DEFAULT_GPU_BATCH,
) -> list[_BlockPredictions]:
    """Phase 4: Run batch GPU predictions for all sequences.

    Orders sequences by strand to efficiency submit batches to the model.

    Args:
        block_seqs: List of extracted sequences from Phase 3.
        ssp_manager: Model manager for running predictions.
        gpu_batch_size: Number of sequences per GPU batch.

    Returns:
        list[_BlockPredictions]: List of prediction results.
    """
    predictions: list[_BlockPredictions] = []

    # Separate by strand to respect the reverse_output contract
    forward = [bs for bs in block_seqs if not bs.reverse_output]
    reverse = [bs for bs in block_seqs if bs.reverse_output]

    for strand_group in (forward, reverse):
        if not strand_group:
            continue

        reverse_output = strand_group[0].reverse_output

        # Interleave: [wt₀, mt₀, wt₁, mt₁, …]
        interleaved: list[str] = []
        for bs in strand_group:
            interleaved.append(bs.wt_seq_clean)
            interleaved.append(bs.mt_seq_clean)

        # Accumulate predictions across GPU mini-batches
        all_acc: list[np.ndarray] = []
        all_dnr: list[np.ndarray] = []

        n_batches = (len(interleaved) + gpu_batch_size - 1) // gpu_batch_size
        pbar_desc = (
            f"Phase 4 — GPU prediction ({'−' if reverse_output else '+'}  strand)"
        )
        for i in tqdm(range(n_batches), desc=pbar_desc, leave=True):
            batch = interleaved[i * gpu_batch_size : (i + 1) * gpu_batch_size]
            acc_batch, dnr_batch = ssp_manager.predict_ssp(
                seqs=batch, reverse_output=reverse_output
            )
            all_acc.extend(acc_batch)
            all_dnr.extend(dnr_batch)

        # De-interleave: even indices = wt, odd = mt
        for idx, bs in enumerate(strand_group):
            predictions.append(
                _BlockPredictions(
                    # SSPScorer inputs
                    chrom_start=int(bs.block.max_start),
                    splice_sites=bs.block.gene_obj.splice_sites,
                    wt_seq=bs.wt_seq,
                    mt_seq=bs.mt_seq,
                    wt_acc=all_acc[2 * idx],
                    wt_dnr=all_dnr[2 * idx],
                    mt_acc=all_acc[2 * idx + 1],
                    mt_dnr=all_dnr[2 * idx + 1],
                    extraction_start=bs.extraction_start,
                    block_type=bs.block.block_type,
                    # Result-row metadata (extracted to avoid pickling HaplotypeBlock)
                    chrom=cast(str, bs.block.chrom),
                    end=int(bs.block.min_end),
                    strand=bs.block.gene_obj.strand,
                    gene_name=bs.block.gene_obj.gene_name,
                    gene_id=bs.block.gene_obj.gene_id,
                    population=bs.block.population,
                    background_id=bs.block.background_id,
                    haplotype_id=bs.block.haplotype_id,
                    block_id=str(bs.block.block_id),
                    block_name=bs.block.name,
                )
            )

    return predictions


# ------------------------------------------------------------------
# Phase 5 — Parallel delta scoring  (ProcessPoolExecutor)
# ------------------------------------------------------------------


def phase5_compute_deltas(
    predictions: list[_BlockPredictions],
    n_workers: int | None = None,
) -> list[dict]:
    """Phase 5: Compute delta scores in parallel.

    Args:
        predictions: Prediction data from Phase 4.
        n_workers: Number of CPU processes to spawn.

    Returns:
        list[dict]: List of result rows for the final output.
    """
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(
            _compute_delta_scores,
            predictions,
            chunksize=_DELTA_CHUNKSIZE,
        )
        for row in tqdm(
            results,
            total=len(predictions),
            desc="Phase 5 — computing delta scores (parallel CPU)",
            leave=True,
        ):
            rows.append(row)
    return rows


# ------------------------------------------------------------------
# Phase 6 — Generate WIG files (ThreadPoolExecutor for I/O)
# ------------------------------------------------------------------


def phase6_generate_wig(
    predictions: list[_BlockPredictions],
    outdir: str,
    n_threads: int = _DEFAULT_IO_THREADS,
) -> None:
    """Phase 6: Generate WIG track files in parallel.

    Args:
        predictions: Prediction data.
        outdir: Output directory.
        n_threads: Number of I/O threads to use.
    """
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        future_map = {
            executor.submit(_generate_wig, outdir, block) for block in predictions
        }
        with tqdm(
            total=len(future_map),
            desc="Phase 6 — generating WIG files (parallel I/O)",
            leave=True,
        ) as pbar:
            for future in as_completed(future_map):
                pbar.update()
