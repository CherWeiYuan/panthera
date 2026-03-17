import logging

from src.panthera.core.runtime import initialize_runtime
from src.panthera.utils.logging_config.py import setup_logging

# Create a module-level logger
logger = logging.getLogger(__name__)


class PantheraOrchestrator:
    def __init__(self, prefix: str, outdir: str, model_type: str, silent: bool):
        self.prefix = prefix
        self.outdir = outdir
        self.model_type = model_type

        # Initialize the logging system
        setup_logging(outdir, prefix, silent)

        # Configure system and hardware
        self.hardware_info = initialize_runtime(silent=silent)

        logger.debug(f"Engine params: prefix={prefix}, model={model_type}")

    def run_survey(self, fasta: str, **kwargs):
        """Orchestrates the haplotype survey pipeline."""
        try:
            # Simulate a step
            logger.debug("Loading genomic indices...")
            # ... scientific logic ...

            if not kwargs.get("phased_vcf"):
                logger.warning(
                    "No VCF provided. Proceeding with GRCh38 reference only."
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
