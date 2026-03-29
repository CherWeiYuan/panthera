"""
Main entry point for Panthera.

Missing features
1. custom background
2. multiprocessing
"""

import time
import sys
import logging
import platform
import resource

import click

# Initialize runtime before tensorflow to suppress tensorflowwarnings
from panthera.utils.runtime import initialize_runtime

initialize_runtime(silent=True, use_mixed_precision=True)

# Import remaining libraries while telling ruff to ignore its checks
from panthera.core.orchestrator import PantheraOrchestrator  # noqa: E402
from panthera.utils.logging_config import setup_logging  # noqa: E402

# Using perf_counter() for high-precision, monotonic time
APP_START_TIME = time.perf_counter()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-p", "--prefix", type=str, default="out")
@click.option("-o", "--outdir", type=str, default="panthera_out")
@click.option(
    "-m", "--model_name", default="modelp", type=click.Choice(["modelp", "spliceai"])
)
@click.option("--silent", is_flag=True, default=False)
@click.pass_context
def cli(ctx, prefix, outdir, model_name, silent):
    # Initialize logging INSIDE the command group so it runs for every command
    setup_logging(outdir="logs", prefix=prefix, silent=silent)

    ctx.obj = PantheraOrchestrator(
        prefix=prefix, outdir=outdir, model_name=model_name, silent=silent
    )


def common_options(f):
    f = click.option("-p", "--prefix", type=str, default="out")(f)
    f = click.option("-o", "--outdir", type=str, default="panthera_out")(f)
    f = click.option(
        "-m",
        "--model_name",
        default="modelp",
        type=click.Choice(["modelp", "spliceai"]),
    )(f)
    f = click.option("--silent", is_flag=True, default=False)(f)
    return f


@cli.command("survey")
@common_options
@click.option(
    "-v",
    "--phased_vcf",
    type=str,
    default=None,
    help="Name of WhatsHap variant call file (.vcf OR .vcf.gz). Use '0|0' "
    "in GT tag to avoid analyzing the variant",
)
@click.option(
    "-t",
    "--tsv",
    type=str,
    default=None,
    help="Name of tab-separated file (.tsv). Mandatory to have at least 4 columns: "
    "chrom (each row with 'chr', e.g. chromosome 1 will be chr1), pos, ref and alt",
)
@click.option(
    "-f", "--fasta", type=str, required=True, help="Name of genomic fasta file."
)
@click.option(
    "--gtf",
    type=str,
    required=True,
    help="Directory and file name of GENCODE GTF",
)
@click.option(
    "-x",
    "--block_extension",
    type=int,
    default=1000,
    help="For each phase set, look at N bases beyond the first and last "
    "variant and consider homozygote variants found as a single haplotype block.",
)
@click.option(
    "-d",
    "--context_dist",
    type=int,
    default=5000,
    metavar="[50-15,000]",
    help="Length of sequence as context. A key factor affecting runtime. "
    "Default of 5,000 refers to the distance of 2500 bp up- and downstream "
    "from the first and last variant.",
)
@click.option(
    "--genetic_background_dir",
    type=str,
    default=None,
    help="Directory to find genetic background VCF files. If not provided, "
    "Panthera will look into panthera/data/genetic_background for the "
    "VCF files.",
)
@click.option(
    "-b",
    "--genetic_background",
    default="NRG",
    type=click.Choice(
        ["BASE", "CUSTOM", "ALL", "NRG", "SUB", "AFR", "AMR", "EAS", "EUR", "SAS"]
    ),
    help="Select genetic background. Choose 'BASE' for GRCh38, 'ALL' for all "
    "population background, 'NRG' for all but the offspring of parent samples, "
    "'SUB' for one individual per superpopulation 'AFR' (African), 'AMR' "
    "(Admixed American), 'EUR' (European), 'EAS' (East Asian), 'SAS' (South Asian)",
)
@click.option(
    "-r",
    "--resolve_variant_conflicts",
    type=bool,
    default=False,
    help="Resolve variant conflicts. If True, target variants supplied in the "
    "TSV or VCF file will be preferentially retained over the genetic "
    "background variants. If False, Panthera will raise error and skip the "
    "affected genetic background for further analysis.",
)
@click.option(
    "-g",
    "--gene_target",
    multiple=True,
    default=(),
    help="(Optional) Name(s) of target gene. Useful to target a specific gene "
    "when multiple genes are sharing the same locus. Example: -g FAS -g ACTA2",
)
@click.option(
    "-k",
    "--custom_background",
    multiple=True,
    default=(),
    help="Names of reference haplotype ID to use as genetic background. Overrides "
    "--genetic_background. Example: -k NA12878 -k NA19240 -k NA19983",
)
@click.option(
    "--generate_wig",
    is_flag=True,
    help="Generate WIG files for IGV visualization of splice site locations.",
)
@click.option(
    "-c",
    "--cpus",
    type=int,
    default=4,
    help="Number of CPU cores or threads to use.",
)
@click.option(
    "-s",
    "--batch_size",
    type=int,
    default=2,
    metavar="[1-512]",
    help="Number of sequences per batch in prediction step. Value can go as high "
    "as memory usage allows (max: 512).",
)
@click.option(
    "-l",
    "--lru_cache_size",
    type=int,
    default=500,
    help="Number of cached predictions. Default is 500.",
)
@click.pass_obj
def survey(orchestrator: PantheraOrchestrator, **kwargs):
    """Bridge to the survey logic."""
    if not kwargs["phased_vcf"] and not kwargs["tsv"]:
        raise click.UsageError("You must provide either --phased_vcf or --tsv.")
    if kwargs["phased_vcf"] and kwargs["tsv"]:
        raise click.UsageError(
            "Parameters --phased_vcf and --tsv are mutually exclusive. Pick one!"
        )

    try:
        orchestrator.run_survey(**kwargs)
    except Exception as e:
        # Error reporting with styled echo (secho)
        # err=True pipes output to STDERR instead of STDOUT
        click.secho(f"Survey failed: {e}", fg="red", err=True)

        # Raise non-zero exit code to indicate error
        raise click.Abort()


@cli.command("isolate")
@common_options
@click.option(
    "-t",
    "--tsv",
    type=str,
    required=True,
    help="Name of tab-separated file (.tsv). Mandatory to have with 4 columns: "
         "chrom, pos, ref, alt.",
)
@click.option(
    "-f", "--fasta", type=str, required=True, help="Name of genomic fasta file."
)
@click.option(
    "-d",
    "--context_dist",
    type=int,
    default=5000,
    metavar="[50-15,000]",
    help="Length of sequence as context. A key factor affecting runtime. "
    "Default of 5,000 refers to the distance of 2500 bp up- and downstream "
    "from the first and last variant.",
)
@click.option(
    "--gtf",
    type=str,
    required=True,
    help="Directory and file name of GENCODE GTF",
)
@click.option(
    "-g",
    "--gene_target",
    required=True,
    help="Name of the only target gene. Example: -g FAS -g ACTA2",
)
@click.option(
    "-v",
    "--variant_target",
    type=str,
    required=True,
    help="Name of target variant to include in every haplotype combination. "
         "Format of input is 'chrom-pos-ref-alt'. Example: -v chr1-123456-A-T.",
)
@click.option(
    "-c",
    "--cpus",
    type=int,
    default=4,
    help="Number of CPU cores/ threads to use. Default is 4.",
)
@click.option(
    "-s",
    "--batch_size",
    type=int,
    default=2,
    metavar="[1-512]",
    help="Number of sequences per batch in prediction step. Value can go as high "
    "as memory usage allows (max: 512).",
)
@click.pass_obj
def isolate(orchestrator: PantheraOrchestrator, **kwargs):
    """Bridge to the isolate logic."""
    try:
        orchestrator.run_isolate(**kwargs)
    except Exception as e:
        # Enterprise-level error reporting
        click.secho(f"Isolate failed: {e}", fg="red", err=True)
        raise click.Abort()


@cli.command("query_fasta")
@common_options
@click.option(
    "-f",
    "--fasta",
    type=str,
    required=True,
    help="Name of query fasta file. Include the DNA or RNA sequences "
    "(nucleotide T/U does not matter) in 5' -> 3' direction. Use .fasta or .fa suffix.",
)
@click.pass_obj
def query_fasta(orchestrator: PantheraOrchestrator, **kwargs):
    """Splice site prediction on a fasta."""
    try:
        orchestrator.query_fasta(**kwargs)
    except Exception as e:
        # Enterprise-level error reporting
        click.secho(f"Query fasta failed: {e}", fg="red", err=True)
        raise click.Abort()


@cli.command("query_genomic_range")
@common_options
@click.option(
    "-f",
    "--fasta",
    type=str,
    required=True,
    help="Name of query fasta file. Include only the RNA sequences (nucleotide "
    "T/U does not matter) in 5' -> 3' direction. Use .fasta or .fa suffix. "
    "Multiple entries in one file is accepted. Chromosome in WIG is the "
    "header of each sequence. (i.e. >header_name).",
)
@click.option(
    "--specify_genomic_range",
    type=str,
    required=True,
    help="Genomic region specified in the string format 'chr-start-end-strand', "
    "where strand is either plus or minus, such as 'chr3:9,866,710-9,880,255-minus' "
    "or 'chr3-9,866,710-9,880,255-minus' with or without commas in the genomic coodinates. "
    "Generates WIG output.",
)
@click.option(
    "-a",
    "--add_mutation",
    type=str,
    default=None,
    help="Specify a mutation with 'chr-pos-ref-alt' 'chr1-300-A-T'. "
    "Mutation must be on the plus strand of the genome.",
)
@click.pass_obj
def query_genomic_range(orchestrator: PantheraOrchestrator, **kwargs):
    """Splice site prediction on a genomic region."""
    try:
        orchestrator.query_genomic_range
    except Exception as e:
        # Enterprise-level error reporting
        click.secho(f"Query genomic range failed: {e}", fg="red", err=True)
        raise click.Abort()


def main():
    # Initialize logging
    logger = logging.getLogger("panthera.main")

    # Run click CLI
    try:
        # Use standalone_mode=False so Click returns here instead of exiting
        cli(standalone_mode=False)
    except click.exceptions.Abort:
        logger.error("Operation aborted by user or error.")
        sys.exit(1)
    except click.exceptions.ClickException as e:
        logger.error(f"Click error: {e.format_message()}")
        sys.exit(e.exit_code)
    except Exception as e:
        logger.exception(f"Application encountered a fatal error: {e}")
        sys.exit(1)
    finally:
        # 1. Calculate Runtime
        total_duration = time.perf_counter() - APP_START_TIME

        # 2. Calculate Peak RAM
        # ru_maxrss returns the maximum resident set size used
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # OS-specific conversion to Megabytes
        if platform.system() == "Darwin":  # macOS
            peak_mem_mb = usage / (1024 * 1024)
        else:  # Linux
            peak_mem_mb = usage / 1024

        # 3. Format Time
        if total_duration < 60:
            time_str = f"{total_duration:.2f}s"
        else:
            minutes, seconds = divmod(total_duration, 60)
            time_str = f"{int(minutes)}m {seconds:.2f}s"

        # 4. Final Enterprise Log Entry
        logger.info("-" * 40)
        logger.info("PROCESS SUMMARY")
        logger.info(f"Total Runtime: {time_str}")
        logger.info(f"Peak Memory:   {peak_mem_mb:.2f} MB")
        logger.info("-" * 40)


if __name__ == "__main__":
    main()
