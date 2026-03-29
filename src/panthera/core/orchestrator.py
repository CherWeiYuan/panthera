import logging
from typing import Literal

from os import makedirs
import pandas as pd

from panthera.core.bio.gene import GTFParser
from panthera.core.bio.io import read_variants
from panthera.core.bio.parse_bg_vcf import BgVcfManager
from panthera.core.bio.split_by_haplotype import split_by_haplotype

from panthera.core.ssp.ssp_manager import SSPManager

from panthera.utils.get_unique_df import get_unique_df

from panthera.utils.constants import hap_dict


# Set up module-level logging
logger = logging.getLogger(__name__)


class PantheraOrchestrator:
    prefix: str
    outdir: str
    model_name: Literal["modelp", "spliceai"]

    def __init__(
        self,
        prefix: str,
        outdir: str,
        model_name: Literal["modelp", "spliceai"],
        silent: bool,
    ):
        self.prefix = prefix
        self.outdir = outdir
        self.model_name = model_name

        logger.debug(f"Engine params: prefix={prefix}, model={model_name}")

        # Create output directory if it doesn't exist
        makedirs(outdir, exist_ok=True)

    def run_survey(self, **kwargs) -> None:
        """
        Orchestrates the haplotype survey pipeline.
        """
        from panthera.core.pipelines.survey import (
            phase1_build_blocks,
            phase2_add_background,
            phase3_extract_sequences,
            phase4_batch_predict,
            phase5_compute_deltas,
            phase6_generate_wig,
        )

        try:
            logger.info("---- Panthera SURVEY ----")

            # ----------------------------------------------------------------
            # Initialisation
            # ----------------------------------------------------------------
            gtf_dict = GTFParser(gtf_file=kwargs["gtf"]).get_gtf_dict()
            bg_vcf_manager = BgVcfManager(external_dir=kwargs["genetic_background_dir"])
            ssp_manager = SSPManager(
                model_name=self.model_name,
                batch_size=kwargs["batch_size"],
                max_cache_size=kwargs["lru_cache_size"],
            )

            gb_group_name = kwargs["genetic_background"]
            gb_samples = (
                hap_dict[gb_group_name]
                if gb_group_name != "CUSTOM"
                else kwargs["custom_background"]
            )

            # ----------------------------------------------------------------
            # Phase 1 — Build haplotype blocks
            # ----------------------------------------------------------------
            input_file = kwargs["phased_vcf"] or kwargs["tsv"]
            vdf = read_variants(input_file)
            contiguous_vdfs = get_unique_df(list(split_by_haplotype(vdf)))
            vdf = None  # free memory early

            haplotype_blocks, single_variant_blocks = phase1_build_blocks(
                contiguous_vdfs=contiguous_vdfs,
                gtf_dict=gtf_dict,
                block_extension=kwargs["block_extension"],
            )
            logger.info(
                "Phase 1 complete: %d haplotype blocks, %d single-variant blocks.",
                len(haplotype_blocks),
                len(single_variant_blocks),
            )

            # ----------------------------------------------------------------
            # Phase 2 — Add genetic background (parallel I/O)
            # ----------------------------------------------------------------
            target_background_blocks: list = []
            if gb_group_name != "BASE":
                target_background_blocks = phase2_add_background(
                    haplotype_blocks=haplotype_blocks,
                    gb_samples=gb_samples,
                    gb_group_name=gb_group_name,
                    bg_vcf_manager=bg_vcf_manager,
                    resolve_conflicts=kwargs["resolve_variant_conflicts"],
                    n_threads=kwargs["cpus"],
                )
            logger.info(
                "Phase 2 complete: %d background blocks added.",
                len(target_background_blocks),
            )

            # Consolidate — background blocks first so they follow their
            # parent haplotype block when sorted by chrom in Phase 3
            all_blocks = (
                haplotype_blocks + single_variant_blocks + target_background_blocks
            )

            # ----------------------------------------------------------------
            # Phase 3 — Extract sequences (chrom-sorted, single FASTA pass)
            # ----------------------------------------------------------------
            block_seqs = phase3_extract_sequences(
                all_blocks=all_blocks,
                ssp_manager=ssp_manager,
                genome_path=kwargs["fasta"],
                context_dist=kwargs["context_dist"],
            )
            logger.info(
                "Phase 3 complete: %d sequence pairs ready for prediction.",
                len(block_seqs),
            )

            # ----------------------------------------------------------------
            # Phase 4 — Batch GPU prediction
            # ----------------------------------------------------------------
            predictions = phase4_batch_predict(
                block_seqs=block_seqs,
                ssp_manager=ssp_manager,
                gpu_batch_size=kwargs["batch_size"],
            )
            logger.info("Phase 4 complete: %d predictions generated.", len(predictions))

            # ----------------------------------------------------------------
            # Phase 5 — Parallel delta scoring
            # ----------------------------------------------------------------
            summary_df_rows = phase5_compute_deltas(
                predictions=predictions,
                n_workers=kwargs["cpus"],
            )
            logger.info(
                "Phase 5 complete: %d delta-score rows computed.", len(summary_df_rows)
            )

            # ----------------------------------------------------------------
            # Phase 6 — Generate WIG files
            # ----------------------------------------------------------------
            if kwargs["generate_wig"]:
                phase6_generate_wig(
                    predictions=predictions,
                    outdir=self.outdir,
                )
                logger.info("Phase 6 complete: WIG files generated.")

            # ----------------------------------------------------------------
            # Save results
            # ----------------------------------------------------------------
            out_path = f"{self.outdir}/survey_results.tsv"
            pd.DataFrame(summary_df_rows).to_csv(
                path_or_buf=out_path, sep="\t", index=False
            )
            logger.info("Survey complete. Results written to %s", out_path)

        except Exception:
            # We log the full stack trace to the file, but a clean message to console
            logger.exception("A fatal error occurred during the survey process.")
            raise

    def run_isolate(self, **kwargs):
        """Orchestrates the variant isolation pipeline."""
        logger.info("----Panthera ISOLATE----")

        from panthera.core.pipelines.isolate import (
            phase1_create_haplotype_combinations
        )

        from panthera.core.pipelines.survey import (
            phase3_extract_sequences,
            phase4_batch_predict,
            phase5_compute_deltas
        )

        try:
            logger.info("---- Panthera ISOLATE ----")

            # ----------------------------------------------------------------
            # Initialisation
            # ----------------------------------------------------------------
            gtf_dict = GTFParser(gtf_file=kwargs["gtf"]).get_gtf_dict()
            ssp_manager = SSPManager(
                model_name=self.model_name,
                batch_size=kwargs["batch_size"],
                max_cache_size=kwargs["lru_cache_size"],
            )

            # ----------------------------------------------------------------
            # Phase 1 — Build haplotype blocks
            # ----------------------------------------------------------------
            input_file = kwargs["phased_vcf"] or kwargs["tsv"]
            vdf = read_variants(input_file)

            haplotype_blocks = phase1_create_haplotype_combinations(
                vdf=vdf,
                gtf_dict=gtf_dict,
                gene_target=kwargs["gene_target"],
                variant_target=kwargs["variant_target"],
            )
            logger.info(
                "Phase 1 complete: %d haplotype blocks.",
                len(haplotype_blocks),
            )

            # ----------------------------------------------------------------
            # Phase 3 — Extract sequences (chrom-sorted, single FASTA pass)
            # ----------------------------------------------------------------
            block_seqs = phase3_extract_sequences(
                all_blocks=all_blocks,
                ssp_manager=ssp_manager,
                genome_path=kwargs["fasta"],
                context_dist=kwargs["context_dist"],
            )
            logger.info(
                "Phase 3 complete: %d sequence pairs ready for prediction.",
                len(block_seqs),
            )

            # ----------------------------------------------------------------
            # Phase 4 — Batch GPU prediction
            # ----------------------------------------------------------------
            predictions = phase4_batch_predict(
                block_seqs=block_seqs,
                ssp_manager=ssp_manager,
                gpu_batch_size=kwargs["batch_size"],
            )
            logger.info("Phase 4 complete: %d predictions generated.", len(predictions))

            # ----------------------------------------------------------------
            # Phase 5 — Parallel delta scoring
            # ----------------------------------------------------------------
            summary_df_rows = phase5_compute_deltas(
                predictions=predictions,
                n_workers=kwargs["cpus"],
            )
            logger.info(
                "Phase 5 complete: %d delta-score rows computed.", len(summary_df_rows)
            )

            # ----------------------------------------------------------------
            # Save results
            # ----------------------------------------------------------------
            out_path = f"{self.outdir}/survey_results.tsv"
            pd.DataFrame(summary_df_rows).to_csv(
                path_or_buf=out_path, sep="\t", index=False
            )
            logger.info("Survey complete. Results written to %s", out_path)



    def query_fasta(self, fasta_path: str):
        """Splice site prediction logic."""
        logger.info("----Panthera QUERY FASTA----")
        pass

    def query_genomic_range(self, fasta_path: str):
        """Splice site prediction logic."""
        logger.info("----Panthera QUERY GENOMIC RANGE----")
        pass
