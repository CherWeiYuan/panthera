"""Panthera MCP Server.

Exposes Panthera pipelines as MCP tools using FastMCP.

NOTE: Parameters are declared as flat, annotated arguments (not as a single
nested BaseModel) so that FastMCP generates a flat JSON Schema for each tool.
MCP clients (including LLMs) expect tools to accept top-level properties, not
a single nested 'args' object.

Pydantic validation is still applied automatically by FastMCP for each
annotated parameter (type coercion, ge/le bounds, Literal choices, etc.).
"""

from typing import Annotated, List, Literal, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from panthera.core.orchestrator import PantheraOrchestrator

mcp = FastMCP("Panthera", instructions="Panthera Splice Site Prediction MCP Server")

# ---------------------------------------------------------------------------
# Shared type aliases
# ---------------------------------------------------------------------------

ModelName = Literal["modelp", "spliceai"]
GeneticBackground = Literal[
    "BASE", "CUSTOM", "ALL", "NRG", "SUB", "AFR", "AMR", "EAS", "EUR", "SAS"
]


# ---------------------------------------------------------------------------
# survey
# ---------------------------------------------------------------------------


@mcp.tool(
    name="survey",
    description="""
        Run the survey pipeline to identify a spliceogenic variant, or a set of 
        variants (haplotype). These genetic variants on the human reference
        genome can be specified in using a VCF file or a TSV file.

        The VCF file must first be phased by WhatsHap before using as input to
        Panthera's survey. For example:

        ```bash
        whatshap phase \
            --indels \
            --distrust-genotypes \
            --include-homozygous \
            --merge-reads \
            --internal-downsampling 18 \
            --tag PS \
            -o <phased_vcf_file> \
            --reference genome/GRCh38.p14.genome.fasta \
            <raw_vcf_file> \
            <sorted_bam_file>
        ```

        The MCP server here prefers to accept a TSV file as input. The TSV file
        must have at least 4 columns with the following headers: 
        - chrom (each row with 'chr', e.g. chromosome 1 will be chr1),
        - pos (position of the variant),
        - ref (reference allele),
        - alt (alternative allele).

        Another important parameter is genetic_background. I recommend using 
        'NRG' for all available non-redundant genomes.

        For context_dist, I recommend 5000 if using --model_name modelp, and 
        15000 if using --model_name spliceai.

        The following is the suggested run parameters:
        ```bash
        uv run panthera survey \
            --tsv <tsv_file> \
            --fasta src/panthera/data/genome/GRCh38.p14.genome.fasta \
            --gtf src/panthera/data/genome/gencode.v49.annotation.gtf.gz \
            --block_extension 1000 \
            --context_dist 5000 \
            --genetic_background NRG \
            --batch_size 1 \
            --cpus 6 \
            --model_name modelp \
            --generate_wig \
            --outdir <outdir> \
            --prefix <prefix>
        ```

        Following the execution, the output will be saved in the output 
        directory (specified in the `--outdir`) in `survey_results.tsv`:
        ```tsv
        | Column header | Description |
        |-------|-----|
        | chrom | Chromosome |
        | start | Start genomic coordinate |
        | end | End genomic coordinate |
        | strand | Plus or minus strand of gene |
        | gene_name | Gene name |
        | gene_id | Gene ID according to GTF file |
        | population | African ('AFR'), Admixed American ('AMR'), East Asian ('EAS'), European ('EUR') or South Asian ('SAS') |
        | genetic_background | ID of individual in the population |
        | haplotype_index | Haplotype A or B representing each chromosome in the human individual |
        | block_ID | ID of the continuous haplotype block |
        | block_type | Type of the block (e.g., 'HAPLOTYPE' or 'SINGLE_VARIANT') |
        | block_variants | Variants in the haplotype block |
        | raw_delta_pos | Positions in the sequence with the highest raw delta |
        | masked_delta_pos | Genomic coordinates of the highest masked delta score |
        | raw_delta | Raw delta score |
        | masked_delta | Highest masked delta score. Splice site probability increase at non-splice sites in the GTF file, or splice site probability decrease at known splice sites. Otherwise, the masked score is zero. |
        ```

        A masked delta score of at least 0.1 is suspected to be spliceogenic.
        The variants in block_variants are in the format of chrom-pos-ref-alt and
        each variant is separated by '.', e.g., "chr3-37007584-C-G.chr3-37007718-G-A".

        If there are multiple variants in a haplotype block, suggest the user
        to run panthera isolate to identify the minimal set of variants that
        is causal to the predicted spliceogenicity.
        
        For any spliceogenic variants identified, suggest the user to perform
        minigene assays in the wet lab to validate the predicted spliceogenicity.
        """,
)
def survey(
    fasta: Annotated[str, Field(description="Path to the genomic FASTA file.")],
    gtf: Annotated[str, Field(description="Path to the GENCODE GTF annotation file.")],
    phased_vcf: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "Path to a WhatsHap phased VCF/VCF.GZ file. "
                "Mutually exclusive with --tsv. Provide exactly one."
            ),
        ),
    ] = None,
    tsv: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "Path to a tab-separated variants file (chrom, pos, ref, alt). "
                "Mutually exclusive with --phased_vcf. Provide exactly one."
            ),
        ),
    ] = None,
    prefix: Annotated[
        str, Field(default="out", description="Prefix for output file names.")
    ] = "out",
    outdir: Annotated[
        str, Field(default="panthera_out", description="Output directory path.")
    ] = "panthera_out",
    model_name: Annotated[
        ModelName,
        Field(default="modelp", description="Splice-site model to use."),
    ] = "modelp",
    block_extension: Annotated[
        int,
        Field(
            default=1000,
            ge=0,
            description=(
                "Bases beyond the first/last variant to extend each haplotype block "
                "when searching for homozygous variants."
            ),
        ),
    ] = 1000,
    context_dist: Annotated[
        int,
        Field(
            default=5000,
            ge=50,
            le=15000,
            description=(
                "Total sequence context length in bp (2500 bp up- and downstream "
                "of the first/last variant). Range: 50–15 000."
            ),
        ),
    ] = 5000,
    genetic_background_dir: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "Directory of genetic-background VCF files. "
                "Defaults to the bundled panthera/data/genetic_background_vcf."
            ),
        ),
    ] = None,
    genetic_background: Annotated[
        GeneticBackground,
        Field(
            default="NRG",
            description=(
                "Population background preset. BASE=GRCh38 reference only, "
                "ALL=all backgrounds, NRG=all non-offspring, SUB=one per "
                "superpopulation, AFR/AMR/EAS/EUR/SAS=individual superpopulations, "
                "CUSTOM=use custom_background IDs."
            ),
        ),
    ] = "NRG",
    resolve_variant_conflicts: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "When True, target variants take priority over background variants "
                "at the same locus. When False, conflicting loci raise an error."
            ),
        ),
    ] = False,
    gene_target: Annotated[
        Optional[List[str]],
        Field(
            default=None,
            description=(
                "One or more gene names to target. Useful when multiple genes "
                "share a locus (e.g. ['FAS', 'ACTA2'])."
            ),
        ),
    ] = None,
    custom_background: Annotated[
        Optional[List[str]],
        Field(
            default=None,
            description=(
                "Reference haplotype IDs to use as the genetic background when "
                "genetic_background='CUSTOM' (e.g. ['NA12878', 'NA19240'])."
            ),
        ),
    ] = None,
    generate_wig: Annotated[
        bool,
        Field(
            default=False,
            description="Generate WIG files for IGV visualization.",
        ),
    ] = False,
    cpus: Annotated[
        int,
        Field(default=4, ge=1, description="Number of CPU cores/threads."),
    ] = 4,
    batch_size: Annotated[
        int,
        Field(
            default=2,
            ge=1,
            le=512,
            description="Sequences per GPU batch. Range: 1–512.",
        ),
    ] = 2,
    lru_cache_size: Annotated[
        int,
        Field(default=500, ge=1, description="Number of cached predictions."),
    ] = 500,
) -> str:
    """Run the survey pipeline for large-scale variant screening.

    Provide exactly one of phased_vcf or tsv as the variant input source.
    """
    if not phased_vcf and not tsv:
        return "Error: You must provide either phased_vcf or tsv."
    if phased_vcf and tsv:
        return "Error: phased_vcf and tsv are mutually exclusive. Provide exactly one."

    orchestrator = PantheraOrchestrator(
        prefix=prefix, outdir=outdir, model_name=model_name, silent=True
    )

    kwargs = {
        "fasta": fasta,
        "gtf": gtf,
        "phased_vcf": phased_vcf,
        "tsv": tsv,
        "block_extension": block_extension,
        "context_dist": context_dist,
        "genetic_background_dir": genetic_background_dir,
        "genetic_background": genetic_background,
        "resolve_variant_conflicts": resolve_variant_conflicts,
        "gene_target": tuple(gene_target) if gene_target else (),
        "custom_background": tuple(custom_background) if custom_background else (),
        "generate_wig": generate_wig,
        "cpus": cpus,
        "batch_size": batch_size,
        "lru_cache_size": lru_cache_size,
    }

    try:
        orchestrator.run_survey(**kwargs)
    except Exception as e:
        return f"Survey failed: {e}"

    return f"Survey complete. Results saved to {outdir}/"


# ---------------------------------------------------------------------------
# isolate
# ---------------------------------------------------------------------------


@mcp.tool(
    name="isolate",
    description="""
        Run the isolate pipeline to identify the minimal set of variants that
        is causal to the predicted spliceogenicity.

        A spliceogenic haplotype block consists of multiple variants but not all 
        variants are necessarily drivers or modifiers. To identify the causal 
        variants, run Panthera ISOLATE on the tab-separated values (TSV) file of 
        variants. 
        
        The TSV can be obtained from Panthera SURVEY or created manually on a 
        text file with 4 tab-separated columns:
            - chrom (each row with 'chr', e.g. chromosome 1 will be chr1),
            - pos (position of the variant),
            - ref (reference allele),
            - alt (alternative allele).

        Use the `-v` / `--variant_target` toggle to specify the target variant 
        that must appear in every combination (format: `chrom-pos-ref-alt`).
        This is usually the variant suspected to be the driver of spliceogenicity 
        while the other variants are potential modifiers. A driver mutation,
        when alone, will generate a non-zero masked delta score when used as
        input to `panthera survey`.

        The output file `isolate_results.tsv` will contain the following columns:
        ```tsv
        | Column header | Description |
        |-------|-----|
        | chrom | Chromosome |
        | start | Start genomic coordinate |
        | end | End genomic coordinate |
        | strand | Plus or minus strand of gene |
        | gene_name | Gene name |
        | gene_id | Gene ID according to GTF file |
        | population | African ('AFR'), Admixed American ('AMR'), East Asian ('EAS'), European ('EUR') or South Asian ('SAS') |
        | genetic_background | ID of individual in the population |
        | haplotype_index | Haplotype A or B representing each chromosome in the human individual |
        | block_ID | ID of the continuous haplotype block |
        | block_type | Type of the block (e.g., 'HAPLOTYPE' or 'SINGLE_VARIANT') |
        | block_variants | Variants in the haplotype block |
        | raw_delta_pos | Positions in the sequence with the highest raw delta |
        | masked_delta_pos | Genomic coordinates of the highest masked delta score |
        | raw_delta | Raw delta score |
        | masked_delta | Highest masked delta score. Splice site probability increase at non-splice sites in the GTF file, or splice site probability decrease at known splice sites. Otherwise, the masked score is zero. |
        ```
        
        You can find the variants in each combination under the column 
        "block_variants" of the output file (`isolate_results.tsv`) in the output 
        directory, alongside their respective delta scores.
        
        The smallest combination of variants with the high delta scores are the 
        likely causal variants.
        """,
)
def isolate(
    tsv: Annotated[
        str,
        Field(
            description=(
                "Path to a tab-separated variants file (chrom, pos, ref, alt). "
                "Must contain all variants in the target locus."
            )
        ),
    ],
    fasta: Annotated[str, Field(description="Path to the genomic FASTA file.")],
    gtf: Annotated[str, Field(description="Path to the GENCODE GTF annotation file.")],
    gene_target: Annotated[
        str,
        Field(
            description=(
                "Name of the single target gene to analyse (e.g. 'FAS'). "
                "Use the survey tool when targeting multiple genes."
            )
        ),
    ],
    variant_target: Annotated[
        str,
        Field(
            description=(
                "The variant that must be present in every haplotype combination. "
                "Format: 'chrom-pos-ref-alt' (e.g. 'chr1-123456-A-T')."
            )
        ),
    ],
    prefix: Annotated[
        str, Field(default="out", description="Prefix for output file names.")
    ] = "out",
    outdir: Annotated[
        str, Field(default="panthera_out", description="Output directory path.")
    ] = "panthera_out",
    model_name: Annotated[
        ModelName,
        Field(default="modelp", description="Splice-site model to use."),
    ] = "modelp",
    context_dist: Annotated[
        int,
        Field(
            default=5000,
            ge=50,
            le=15000,
            description="Total sequence context length in bp. Range: 50–15 000.",
        ),
    ] = 5000,
    cpus: Annotated[
        int, Field(default=4, ge=1, description="Number of CPU cores/threads.")
    ] = 4,
    batch_size: Annotated[
        int,
        Field(
            default=2,
            ge=1,
            le=512,
            description="Sequences per GPU batch. Range: 1–512.",
        ),
    ] = 2,
) -> str:
    """Run the isolate pipeline for targeted haplotype combination analysis."""
    orchestrator = PantheraOrchestrator(
        prefix=prefix, outdir=outdir, model_name=model_name, silent=True
    )

    kwargs = {
        "tsv": tsv,
        "fasta": fasta,
        "gtf": gtf,
        "gene_target": gene_target,
        "variant_target": variant_target,
        "context_dist": context_dist,
        "cpus": cpus,
        "batch_size": batch_size,
    }

    try:
        orchestrator.run_isolate(**kwargs)
    except Exception as e:
        return f"Isolate failed: {e}"

    return f"Isolate complete. Results saved to {outdir}/"


# ---------------------------------------------------------------------------
# query_fasta
# ---------------------------------------------------------------------------


@mcp.tool(
    name="query_fasta",
    description="""
        Run the query_fasta pipeline to generate a WIG file that can be used
        in IGV to visualize the splice site probabilities of the input sequences.

        Ensure the input FASTA sequence is provided in the 5' -> 3' direction.
        """,
)
def query_fasta(
    fasta: Annotated[
        str,
        Field(
            description=(
                "Path to the input FASTA file. Each entry is scored independently. "
                "Use .fasta or .fa extension. Ensure the sequence is provided "
                "in the 5' -> 3' direction."
            )
        ),
    ],
    prefix: Annotated[
        str, Field(default="out", description="Prefix for output file names.")
    ] = "out",
    outdir: Annotated[
        str, Field(default="panthera_out", description="Output directory path.")
    ] = "panthera_out",
    model_name: Annotated[
        ModelName,
        Field(default="modelp", description="Splice-site model to use."),
    ] = "modelp",
) -> str:
    """Predict splice site probabilities for sequences in a FASTA file."""
    orchestrator = PantheraOrchestrator(
        prefix=prefix, outdir=outdir, model_name=model_name, silent=True
    )

    try:
        orchestrator.query_fasta(fasta=fasta)
    except Exception as e:
        return f"Query fasta failed: {e}"

    return f"Query fasta complete. Results saved to {outdir}/"


# ---------------------------------------------------------------------------
# query_genomic_range
# ---------------------------------------------------------------------------


@mcp.tool(
    name="query_genomic_range",
    description="""
        Run the query_genomic_range pipeline to generate a WIG file that can be used
        in IGV to visualize the splice site probabilities of the input sequences.

        While query_fasta uses a FASTA input, this tool allows the input of a
        genomic range (e.g., 'chr3:5,667,890-5,677,890-plus') and the sequence
        will be extracted from the genomic FASTA file for splice site prediction.
        """,
)
def query_genomic_range(
    fasta: Annotated[str, Field(description="Path to the genomic FASTA file.")],
    genomic_range: Annotated[
        str,
        Field(
            description=(
                "Genomic region in 'chr-start-end-strand' format. Commas in "
                "coordinates are accepted. Strand is 'plus' or 'minus'. "
                "Example: 'chr3:9,866,710-9,880,255-minus'."
            )
        ),
    ],
    prefix: Annotated[
        str, Field(default="out", description="Prefix for output file names.")
    ] = "out",
    outdir: Annotated[
        str, Field(default="panthera_out", description="Output directory path.")
    ] = "panthera_out",
    model_name: Annotated[
        ModelName,
        Field(default="modelp", description="Splice-site model to use."),
    ] = "modelp",
) -> str:
    """Predict splice site probabilities for a specific genomic region."""
    orchestrator = PantheraOrchestrator(
        prefix=prefix, outdir=outdir, model_name=model_name, silent=True
    )

    try:
        orchestrator.query_genomic_range(fasta=fasta, genomic_range=genomic_range)
    except Exception as e:
        return f"Query genomic range failed: {e}"

    return f"Query genomic range complete. Results saved to {outdir}/"
