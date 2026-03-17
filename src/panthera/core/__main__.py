"""
Main entry point for Panthera.
"""

import click

from src.panthera.core.orchestrator import PantheraOrchestrator


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-p", "--prefix", type=str, default="out", help="Output file prefix")
@click.option(
    "-o", "--outdir", type=str, default="panthera_out", help="Output directory"
)
@click.option(
    "-m",
    "--model_type",
    default="modelp",
    type=click.Choice(["modelp", "spliceai", "cispliceai"], case_sensitive=False),
    help="Prediction model engine. Model P works best for exon/shallow intron, "
    "SpliceAI for deep introns and CI-SpliceAI is provided as an orthogonal model.",
)
@click.option(
    "--silent",
    is_flag=True,
    default=False,
    help="Suppress printing of report into stdout.",
)
@click.pass_context
def cli(ctx, prefix, outdir, model_type, silent):
    """
    PANTHERA: Detects splice haplotypes and predicts splice sites.
    """
    # Initialize a dictionary to store our global options
    ctx.ensure_object(dict)
    ctx.obj["PREFIX"] = prefix
    ctx.obj["OUTDIR"] = outdir
    ctx.obj["MODEL_TYPE"] = model_type
    ctx.obj["SILENT"] = silent


@cli.command("survey")
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
    "-x",
    "--whatshap_extension",
    type=int,
    default=1000,
    help="For each WhatsHap phase set, look at N bases beyond the first and last "
    "variant and consider homozygote variants found as a single haplotype block.",
)
@click.option(
    "-d",
    "--context_dist",
    type=int,
    default=3000,
    metavar="[50-15,000]",
    help="Length of sequence as context. A key factor affecting runtime. "
    "Default of 3,000 refers to the distance of 1500 bp up- and downstream "
    "from the first and last variant.",
)
@click.option(
    "--gtf",
    type=str,
    default="genome/gencode.v46.basic.annotation.gtf",
    help="Directory and file name of GENCODE GTF",
)
@click.option(
    "-r",
    "--ref_haplotypes_dir",
    type=str,
    default="genome/reference_haplotypes",
    help="Directory where reference haplotype VCFs are stored. If None, then "
    "reference haplotype will be only GRCh38.",
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
    "-g",
    "--gene_target",
    multiple=True,
    default=[],
    help="(Optional) Name(s) of target gene. Useful to target a specific gene "
    "when multiple genes are sharing the same locus. Example: -g FAS -g ACTA2",
)
@click.option(
    "-k",
    "--custom_background",
    multiple=True,
    default=[],
    help="Names of reference haplotype ID to use as genetic background. Overrides "
    "--genetic_background. Example: -k NA12878 -k NA19240 -k NA19983",
)
@click.option(
    "--write_full_output",
    is_flag=True,
    help="Export fasta sequences, WIG files files for IGV visualization of splice "
    "site locations, and genetic background variants in TSVs. WARNING: Slows "
    "code by disabling the use of cached predictions.",
)
@click.option(
    "-c",
    "--cores",
    type=int,
    default=0,
    help="Number of CPU cores to use. Default uses all CPU cores available "
    "and has less overhead processing.",
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
def survey(engine: PantheraOrchestrator, **kwargs):
    """Bridge to the survey logic."""
    try:
        engine.run_survey(**kwargs)
    except Exception as e:
        # Error reporting with styled echo (secho)
        # err=True pipes output to STDERR instead of STDOUT
        click.secho(f"Survey failed: {e}", fg="red", err=True)

        # Raise non-zero exit code to indicate error
        raise click.Abort()


@cli.command("isolate")
@click.option(
    "-t",
    "--tsv",
    type=str,
    required=True,
    help="Name of tab-separated file (.tsv). Mandatory to have with 5 columns: "
    "chrom, pos, ref, alt and target_variant. The column target_variant with "
    "cells labelled integer 1 specifies the variant that must appear in every "
    "combination.",
)
@click.option(
    "-f", "--fasta", type=str, required=True, help="Name of genomic fasta file."
)
@click.option(
    "-d",
    "--context_dist",
    type=int,
    default=3000,
    metavar="[50-10,000]",
    help="Length of sequence as context. A key factor affecting runtime. "
    "Default of 3,000 refers to the distance of 1500 bp up- and downstream "
    "from the first and last variant.",
)
@click.option(
    "--gtf",
    type=str,
    default="genome/gencode.v46.basic.annotation.gtf",
    help="Directory and file name of GENCODE GTF",
)
@click.option(
    "-g",
    "--gene_target",
    multiple=True,
    default=[],
    help="(Optional) Name(s) of target gene. Useful to target a specific gene "
    "when multiple genes are sharing the same locus. Example: -g FAS -g ACTA2",
)
@click.option(
    "-c",
    "--cores",
    type=int,
    default=0,
    help="Number of CPU cores to use. Default uses all CPU cores available "
    "and has less overhead processing.",
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
def isolate(engine: PantheraOrchestrator, **kwargs):
    """Bridge to the isolate logic."""
    try:
        engine.run_isolate(**kwargs)
    except Exception as e:
        # Enterprise-level error reporting
        click.secho(f"Isolate failed: {e}", fg="red", err=True)
        raise click.Abort()


@cli.command("query_fasta")
@click.option(
    "-f",
    "--fasta",
    type=str,
    required=True,
    help="Name of query fasta file. Include the DNA or RNA sequences "
    "(nucleotide T/U does not matter) in 5' -> 3' direction. Use .fasta or .fa suffix.",
)
@click.pass_obj
def query_fasta(engine: PantheraOrchestrator, **kwargs):
    """Splice site prediction on a fasta."""
    try:
        engine.query_fasta(**kwargs)
    except Exception as e:
        # Enterprise-level error reporting
        click.secho(f"Query fasta failed: {e}", fg="red", err=True)
        raise click.Abort()


@cli.command("query_genomic_range")
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
@click.pass_context
def query_genomic_range(engine: PantheraOrchestrator, **kwargs):
    """Splice site prediction on a genomic region."""
    try:
        engine.query_genomic_range
    except Exception as e:
        # Enterprise-level error reporting
        click.secho(f"Query genomic range failed: {e}", fg="red", err=True)
        raise click.Abort()


if __name__ == "__main__":
    cli()
