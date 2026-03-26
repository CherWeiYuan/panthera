from copy import deepcopy
import logging
from typing import cast

from panthera.core.bio.blocks import HaplotypeBlock, VariantSchema
from panthera.core.bio.extend_phaseset import extend_phaseset
from panthera.core.bio.gene import find_genes_at_pos, GTFParser
from panthera.core.bio.io import read_variants
from panthera.core.bio.parse_bg_vcf import BgVcfManager, VCFCoordinates
from panthera.core.bio.split_by_haplotype import split_by_haplotype

from panthera.utils.get_unique_df import get_unique_df
from panthera.utils.logging_config import setup_logging

from pandera.typing import DataFrame
from panthera.utils.constants import hap_dict

from panthera.utils.exceptions import (
    DataResolutionError,
    BackgroundConflictError,
    AmbiguousDeletionError,
)

# Set up module-level logging
logger = logging.getLogger(__name__)


class PantheraOrchestrator:
    def __init__(self, prefix: str, outdir: str, model_type: str, silent: bool):
        self.prefix = prefix
        self.outdir = outdir
        self.model_type = model_type

        # Initialize the logging system
        setup_logging(outdir, prefix, silent)

        logger.debug(f"Engine params: prefix={prefix}, model={model_type}")

    def run_survey(self, fasta: str, **kwargs):
        """Orchestrates the haplotype survey pipeline."""
        try:
            # ---- Load auxiliary files ---- #

            # Load GTF dictionary
            gtf_dict = GTFParser(gtf_file=kwargs["gtf"]).get_gtf_dict()

            # Load genetic background VCF directory
            bg_vcf_manager = BgVcfManager(external_dir=kwargs["genetic_background_dir"])

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
            for c_vdf in contiguous_vdfs:
                target_blocks = []  # Only target variants
                target_background_blocks = []  # Target variants + Background

                # Get unique chrom-phase set pairs
                unique_pairs = c_vdf[["chrom", "phase_set"]].drop_duplicates()

                # Iterate through each unique phase set
                for chrom, ps in unique_pairs.itertuples(index=False):
                    # Get unique phase set variants dataframe
                    current_vdf = cast(
                        DataFrame[VariantSchema],
                        c_vdf[(c_vdf.chrom == chrom) & (c_vdf.phase_set == ps)],
                    )

                    current_vdf = cast(
                        DataFrame[VariantSchema],
                        extend_phaseset(
                            current_vdf,
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
                            existing_genes=gene_objs,
                        )
                        gene_objs += gene_obj

                    # ---- Add haplotype blocks ---- #
                    for gene_obj in gene_objs:
                        # Add target haplotype block (only target variants)
                        target_block = HaplotypeBlock(
                            variants_df=current_vdf, gene_obj=gene_obj
                        )
                        target_blocks.append(target_block)

                        # Add target haplotype block (with background variants)
                        if kwargs["genetic_background"] != "BASE":
                            # Iterate through Target Haplotype Blocks
                            max_start = target_block.max_start
                            min_end = target_block.min_end
                            coords = VCFCoordinates(
                                chrom=chrom, start=max_start, end=min_end
                            )

                            # Iterate through Genetic Background Samples
                            for gbs in gb_samples:
                                try:
                                    bg_vdf = bg_vcf_manager.fetch_region(
                                        sample_id=gbs, coords=coords
                                    )
                                except DataResolutionError as e:
                                    # Raise error as user wants genetic
                                    # background but the background VCF cannot
                                    # be found
                                    logger.warning(
                                        f"Failed to read background VCF for {gbs}: {e}"
                                        f"Skipping {gbs} as background."
                                    )
                                    raise

                                # Split background variants according to phasing
                                contiguous_bg_vdfs = split_by_haplotype(
                                    cast(DataFrame[VariantSchema], bg_vdf)
                                )
                                bg_vdf = None

                                for hap_id, c_bg_vdf in zip(
                                    ("A", "B"), contiguous_bg_vdfs
                                ):
                                    target_background_block = deepcopy(target_block)
                                    try:
                                        # Add genetic background variants to target haplotype block
                                        # Automatically checks for variant conflicts
                                        target_background_block.add_background_variants(
                                            background_df=c_bg_vdf,
                                            population=gb_group_name,
                                            background_id=gbs,
                                            haplotype_id=hap_id,
                                            resolve_conflicts=kwargs[
                                                "resolve_variant_conflicts"
                                            ],
                                        )

                                        # Check for ambiguous deletions
                                        target_background_block._check_deletion_validity()

                                    except BackgroundConflictError as e:
                                        # If resolve_conflicts is False when conflicts are found,
                                        # warn and skip background
                                        logger.warning(
                                            f"Failed to read background VCF for {gbs}: {e}"
                                            f"Skipping {gbs} as background."
                                        )

                                    except AmbiguousDeletionError as e:
                                        # Ambiguous deletions are not allowed,
                                        # warn and skip background
                                        logger.warning(
                                            f"Deletion variant deleted positions with other variants for {gbs}: {e}"
                                            f"Skipping {gbs} as background."
                                        )

                                    # Append Target + Background Haplotype Block
                                    target_background_blocks.append(
                                        target_background_block
                                    )

        except Exception:
            # We log the full stack trace to the file, but a clean message to console
            logger.exception("A fatal error occurred during the survey process.")
            raise

    def run_isolate(self, **kwargs):
        """Orchestrates the variant isolation pipeline."""
        pass

    def query_fasta(self, fasta_path: str):
        """Splice site prediction logic."""
        pass

    def query_genomic_range(self, fasta_path: str):
        """Splice site prediction logic."""
        pass
