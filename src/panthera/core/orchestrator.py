from copy import deepcopy
import logging
from typing import cast
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from panthera.core.bio.blocks import HaplotypeBlock, VariantSchema
from panthera.core.bio.extend_phaseset import extend_phaseset
from panthera.core.bio.gene import find_genes_at_pos, GTFParser
from panthera.core.bio.io import read_variants
from panthera.core.bio.parse_bg_vcf import BgVcfManager, VCFCoordinates
from panthera.core.bio.parse_genome import GenomeParser
from panthera.core.bio.split_by_haplotype import split_by_haplotype

from panthera.core.ssp.ssp_manager import SSPManager
from panthera.core.ssp.calc_delta import SSPScorer

from panthera.utils.get_unique_df import get_unique_df

from pandera.typing import DataFrame
from panthera.utils.constants import hap_dict

from panthera.utils.exceptions import (
    DataResolutionError,
    BackgroundConflictError,
    AmbiguousDeletionError
)

# Set up module-level logging
logger = logging.getLogger(__name__)


class PantheraOrchestrator:
    def __init__(self, prefix: str, outdir: str, model_name: str, silent: bool):
        self.prefix = prefix
        self.outdir = outdir
        self.model_name = model_name

        logger.debug(f"Engine params: prefix={prefix}, model={model_name}")

    def run_survey(self, **kwargs):
        """Orchestrates the haplotype survey pipeline."""
        try:
            logger.info("----Panthera SURVEY----")

            # ---- Load auxiliary files ---- #

            # Load GTF dictionary
            gtf_dict = GTFParser(gtf_file=kwargs["gtf"]).get_gtf_dict()

            # Load genetic background VCF directory
            bg_vcf_manager = BgVcfManager(external_dir=kwargs["genetic_background_dir"])

            # ---- Load helper class instances ---- #
            genome_parser = GenomeParser()
            ssp_manager = SSPManager(
                model_name = self.model_name,
                batch_size = kwargs["batch_size"],
                max_cache_size = 500)

            # ---- Genetic background ---- #

            # Get genetic background group name
            # ("BASE", "NRG", "EAS", etc.)
            gb_group_name = kwargs["genetic_background"]

            # Get tuple of individual genetic background name
            # ("NA12878", "NA19240", etc.)
            if gb_group_name != "CUSTOM":
                gb_samples = hap_dict[gb_group_name]  # tuple
            else:
                gb_samples = kwargs["custom_background"]  # tuple

            # ---- Target variants ---- #

            # Load VCF or TSV
            input_file = kwargs["phased_vcf"] or kwargs["tsv"]
            vdf = read_variants(input_file)

            # Split variants according to phasing
            contiguous_vdfs = split_by_haplotype(vdf)
            vdf = None

            # Remove duplicated dataframe
            contiguous_vdfs = get_unique_df(list(contiguous_vdfs))

            # ---- Extract phase sets from a contiguous VDF ---- #
            haplotype_blocks = []
            single_variant_blocks = []

            for c_vdf in contiguous_vdfs:

                # Get unique chrom-phase set pairs
                unique_pairs = c_vdf[["chrom", "phase_set"]].drop_duplicates()

                # Iterate through each unique phase set
                pbar = tqdm(total = len(unique_pairs),
                            position = 0, leave = True,
                            desc="Extracting phase sets into target haplotype blocks")
                for chrom, ps in unique_pairs.itertuples(index=False):
                    # Get unique phase set variants dataframe and
                    # extend phase set to include flanking homozygous variants
                    current_vdf = cast(
                        DataFrame[VariantSchema],
                        extend_phaseset(
                            c_vdf,
                            chrom=chrom,
                            ps_id=ps,
                            ext_len=kwargs["block_extension"],
                        ),
                    )

                    # ---- Get gene objects ---- #
                    # Get gene object at each position in current vdf
                    gene_objs = []
                    for pos in current_vdf.pos.unique():
                        gene_obj = find_genes_at_pos(
                            chrom=chrom,
                            pos=pos,
                            gtf_dict=gtf_dict,
                            # Ensures no duplicate gene objects
                            existing_genes=gene_objs,
                        )
                        gene_objs += gene_obj

                    # ---- Add target haplotype blocks ---- #
                    for gene_obj in gene_objs:
                        # Add target single variant block
                        # Each single variant in the current phase set is a target block
                        variant_dfs = np.array_split(
                            current_vdf, len(current_vdf))
                        for variant_df in variant_dfs:
                            target_single_variant_block = HaplotypeBlock(
                                variants_df=variant_df, gene_obj=gene_obj)
                            
                            target_single_variant_block.population = "BASE"
                            target_single_variant_block.background_id = "BASE"

                            single_variant_blocks.append(
                                target_single_variant_block)

                        # Add target haplotype block 
                        # All target variants in the current phase set
                        target_block = HaplotypeBlock(
                            variants_df=current_vdf, gene_obj=gene_obj
                        )
                        haplotype_blocks.append(target_block)
                    
                    # Reset variables
                    current_vdf, target_block, gene_objs = None, None, None
                    pbar.update()

            # ---- Add genetic background to target haplotype blocks ---- #
            # Note: Only add genetic background variants to target HAPLOTYPE
            #       blocks. Do not add to SINGLE VARIANT blocks.
            target_background_blocks = []
            if kwargs["genetic_background"] != "BASE":
                pbar = tqdm(haplotype_blocks,
                            position = 0, leave = True,
                            desc="Adding genetic background to " +
                                 "target haplotype blocks")
                for block in pbar:
                    # Iterate through Target Haplotype Blocks
                    max_start = block.max_start
                    min_end = block.min_end
                    coords = VCFCoordinates(
                            chrom=block.chrom, start=max_start, end=min_end
                        )
                    # Iterate through Genetic Background Samples
                    for gbs in gb_samples:
                        
                        with warnings.catch_warnings(record=True) as warning_list:
                            try:
                                bg_vdf = bg_vcf_manager.fetch_region(
                                    sample_id=gbs, coords=coords
                                )
                            except DataResolutionError as e:
                                # Raise error as user wants genetic background 
                                # but the background VCF cannot be found
                                logger.warning(
                                    f"Failed to read background VCF for {gbs}: {e}"
                                    f"Skipping {gbs} as background."
                                )
                                raise

                        if len(warning_list) > 0:
                            # Empty background VCF
                            logger.warning(
                                f"No variants found in background VCF for {gbs}: {e}"
                                f"Skipping {gbs} as background."
                            )
                            warning_list = None
                            continue
                        

                        # Split background variants according to phase
                        contiguous_bg_vdfs = split_by_haplotype(
                            cast(DataFrame[VariantSchema], bg_vdf))
                        bg_vdf = None

                        for hap_id, c_bg_vdf in zip(
                            ("A", "B"), contiguous_bg_vdfs
                        ):
                            # Deep copy target haplotype block
                            target_background_block = deepcopy(block)

                            # Add genetic background variants to target 
                            # haplotype block
                            try:
                                target_background_block.add_background_variants(
                                    background_df=c_bg_vdf,
                                    population=gb_group_name,
                                    background_id=gbs,
                                    haplotype_id=hap_id,
                                    resolve_conflicts=kwargs["resolve_variant_conflicts"],
                                )

                            except BackgroundConflictError as e:
                                # If resolve_conflicts is False when conflicts 
                                # are found, warn and skip background
                                logger.warning(
                                    f"Failed to read background VCF for {gbs}"
                                    f": {e}.Skipping {gbs} as background."
                                )

                            except AmbiguousDeletionError as e:
                                # Ambiguous deletions are not allowed,
                                # warn and skip background
                                logger.warning(
                                    "Deletion variant deleted positions with "
                                    f"other variants for {gbs}: {e}. "
                                    f"Skipping {gbs} as background."
                                )

                            # Append Target + Background Haplotype Block
                            target_background_blocks.append(
                                target_background_block
                            )
            
            # Consolidate all haplotype blocks
            haplotype_blocks += single_variant_blocks + target_background_blocks

            # ---- Extract WT and MT sequences for all haplotype blocks ---- #
            # Sort haplotype blocks by chromosome
            haplotype_blocks.sort(key = lambda x: x.chrom)
            previous_chrom = None
            summary_df_rows = []

            for block in haplotype_blocks:
                # Load only one chromosome fasta at a time
                current_chrom = block.chrom
                if current_chrom != previous_chrom:
                    chrom_seq = genome_parser.parse_genome(
                        genome_path = kwargs["fasta"], chrom = current_chrom
                    )[current_chrom]
                    previous_chrom = current_chrom
                
                # Extract WT and MT sequences for all haplotype blocks and
                # update sequence within the block instance
                # Note: wt_seq and mt_seq are extracted from the chromosome
                #       sequence without considering strand direction.
                try:
                    wt_seq, mt_seq = block.extract_seqs(
                        chrom_seq = chrom_seq,
                        extension_len = kwargs["block_extension"]
                        )
                except AmbiguousDeletionError as e:
                    logger.warning(f"Ambiguous deletion error for block {block.name}: {e}. Skipping block.")
                    continue

                # Create clean versions for prediction by removing indel markers
                wt_seq_clean = ssp_manager.remove_indel_markers([wt_seq])[0]
                mt_seq_clean = ssp_manager.remove_indel_markers([mt_seq])[0]
                
                # Predict splice site probabilities
                block_strand = block.gene_obj.strand
                if block_strand == "-":
                    # Reverse complement negative strand for model input
                    wt_seq_clean = ssp_manager.reverse_complement([wt_seq_clean])[0]
                    mt_seq_clean = ssp_manager.reverse_complement([mt_seq_clean])[0]
                    reverse_output = True
                else:
                    reverse_output = False

                # Bulk predict for both sequences in one call
                # predict_ssp returns (acceptor_probs_list, donor_probs_list)
                acc_probs_list, dnr_probs_list = ssp_manager.predict_ssp(
                    seqs = [wt_seq_clean, mt_seq_clean],
                    reverse_output = reverse_output
                )

                # Correct assignment:
                # index 0 refers to wt_seq_clean, index 1 to mt_seq_clean
                block.wt_acc = acc_probs_list[0]
                block.wt_dnr = dnr_probs_list[0]
                block.mt_acc = acc_probs_list[1]
                block.mt_dnr = dnr_probs_list[1]

                # ---- Calculate delta scores ---- #
                # Initialize SSP score instance
                delta_scorer = SSPScorer(
                    chrom_start = block.max_start,
                    splice_sites = block.gene_obj.splice_sites,
                    wt_seq = wt_seq,
                    mt_seq = mt_seq,
                    wt_acc = block.wt_acc,
                    wt_dnr = block.wt_dnr,
                    mt_acc = block.mt_acc,
                    mt_dnr = block.mt_dnr
                )

                # Align splice site probabilities
                delta_scorer.align_prob()

                # Calculate delta scores (output delta scores as numpy array)
                raw_deltas = delta_scorer.calc_raw_deltas()
                masked_deltas = delta_scorer.calc_masked_deltas()

                # Get max delta score (one float value)
                max_raw_delta = round(float(np.max(raw_deltas)), 3)
                max_masked_delta = round(float(np.max(masked_deltas)), 3)

                # Find max delta positions
                max_raw_delta_loc = delta_scorer._find_max_delta_locations(
                    max_deltas = raw_deltas,
                    max_val = max_raw_delta
                )
                max_masked_delta_loc = delta_scorer._find_max_delta_locations(
                    max_deltas = masked_deltas,
                    max_val = max_masked_delta
                )

                # Store results as pre-dataframe rows
                summary_df_rows.append(
                    {
                        "chrom": block.chrom,
                        "start": block.max_start,
                        "end": block.min_end,
                        "strand": block.gene_obj.strand,
                        "gene_name": block.gene_obj.gene_name,
                        "gene_id": block.gene_obj.gene_id,
                        "population": block.population,            # e.g. "EAS"
                        "genetic_background": block.background_id, # e.g. "NA19238"
                        "haplotype_index": block.haplotype_id,     # 'A'/ 'B'
                        "block_ID": block.block_id,
                        "block_variants": block.name,
                        "raw_delta_pos": max_raw_delta_loc,
                        "masked_delta_pos": max_masked_delta_loc,
                        "raw_delta": max_raw_delta,
                        "masked_delta": max_masked_delta,
                    }
                )

            # ---- Create summary dataframe ---- #
            summary_df = pd.DataFrame(summary_df_rows)

            # ---- Save summary dataframe ---- #
            summary_df.to_csv(
                path_or_buf=f"{self.outdir}/survey_results.tsv",
                sep="\t",
                index=False
            )

        except Exception:
            # We log the full stack trace to the file, but a clean message to console
            logger.exception("A fatal error occurred during the survey process.")
            raise

    def run_isolate(self, **kwargs):
        """Orchestrates the variant isolation pipeline."""
        logger.info("----Panthera ISOLATE----")
        pass

    def query_fasta(self, fasta_path: str):
        """Splice site prediction logic."""
        logger.info("----Panthera QUERY FASTA----")
        pass

    def query_genomic_range(self, fasta_path: str):
        """Splice site prediction logic."""
        logger.info("----Panthera QUERY GENOMIC RANGE----")
        pass
