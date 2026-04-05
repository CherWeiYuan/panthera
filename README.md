# Panthera
A toolkit for splice haplotype prediction and validation.
<br />

![Python Version](https://img.shields.io/badge/python-3.10-blue) 
[![Python Tests](https://github.com/CherWeiYuan/panthera/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/CherWeiYuan/panthera/actions/workflows/test.yml)
![PyPI](https://img.shields.io/pypi/v/YOUR_PACKAGE_NAME)
![License](https://img.shields.io/github/license/CherWeiYuan/panthera)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Setting up Panthera

### Required files
Download the [genome folder from Google Drive](https://drive.google.com/drive/folders/1-_7Tl3mVknu1TPKGLl-fIBCkflXFnLxm?usp=sharing).

### Installation

Panthera uses only PyPI libraries. It requires Python 3.10 and can be installed using `pip`.

```bash
pip install --upgrade pip
pip install panthera
```
<br />

## Navigating Panthera's Command-line Interface (CLI)
When in doubt, run the help command in Panthera to access information on subcommands and their parameters.
```bash
panthera --help
```

```
Usage: panthera [OPTIONS] COMMAND [ARGS]...

  Panthera: Splice site probability prediction tool.

Options:
  -p, --prefix TEXT               Prefix string for the output files generated
                                  by Panthera.
  -o, --outdir TEXT               Path to the directory where calculation
                                  results will be saved.
  -m, --model_name [modelp|spliceai]
                                  Specify to use either Panthera ('modelp') or
                                  SpliceAI ('spliceai') as the underlying
                                  neural network.
  --silent                        Suppress terminal output logging except for
                                  critical errors.
  -h, --help                      Show this message and exit.

Commands:
  isolate              Runs the isolate pipeline for targeted haplotype combinations.
  query_fasta          Performs splice site prediction on a user-supplied FASTA file.
  query_genomic_range  Performs splice site prediction on a specific genomic region.
  survey               Runs the survey pipeline for large-scale variant screening.
```
<br />

For help on the subcommands (e.g., the survey module), use:
```bash
panthera survey --help
```
<br />

## [PANTHERA CLI SURVEY] How to predict splice haplotypes in a VCF?
Step 1. Run [WhatsHap](https://whatshap.readthedocs.io/en/latest/guide.html) to phase your VCF
```bash
whatshap phase \
    --indels \
    --distrust-genotypes \
    --include-homozygous \
    --merge-reads \
    --internal-downsampling 18 \
    --tag PS \
    -o example/input/sample.phased.vcf \
    --reference genome/GRCh38.p14.genome.fasta \
    sample.raw.vcf \
    sample.bam
```
<br />

Step 2. Run Panthera SURVEY on phased vcf
```bash
panthera survey \
    --phased_vcf example/input/sample.phased.vcf \
    --fasta genome/GRCh38.p14.genome.fasta \
    --gtf genome/gencode.v47.basic.annotation.gtf \
    --outdir example/output \
    --prefix SURVEY_EXAMPLE
```
For more options, see `panthera survey --help`

<br />

## Output
You can access the output in the output directory (specified in the `--outdir`) in `survey_results.tsv` (for the survey pipeline) or `isolate_results.tsv` (for the isolate pipeline):

| Column header | Description |
|-------|-----|
| chrom | Chromosome |
| start | Start genomic coordinate |
| end | End genomic coordinate |
| strand | Plus or minus strand of gene |
| gene_name | Gene name |
| gene_id | Gene ID according to GTF file |
| population | African ('AFR'), American ('AMR'), East Asian ('EAS'), European ('EUR') or South Asian ('SAS') |
| genetic_background | ID of individual in the population |
| haplotype_index | Haplotype A or B representing each chromosome in the human individual |
| block_ID | ID of the continuous haplotype block |
| block_type | Type of the block (e.g., 'HAPLOTYPE' or 'SINGLE_VARIANT') |
| block_variants | Variants in the haplotype block |
| raw_delta_pos | Positions in the sequence with the highest raw delta |
| masked_delta_pos | Genomic coordinates of the highest masked delta score |
| raw_delta | Raw delta score |
| masked_delta | Highest masked delta score. Splice site probability increase at non-splice sites in the GTF file, or splice site probability decrease at known splice sites. Otherwise, the masked score is zero. |
<br />

## [PANTHERA CLI ISOLATE] How to find causal variants in a haplotype block?
A spliceogenic haplotype block consists of multiple variants but not all variants are necessarily drivers or modifiers.

To identify the causal variants, run Panthera ISOLATE on the tab-separated values (TSV) file of variants. The TSV can be obtained from Panthera SURVEY or created manually on a text file with 4 tab-separated columns: chrom, pos, ref, alt.

Use the `-v` / `--variant_target` toggle to specify the target variant that must appear in every combination (format: `chrom-pos-ref-alt`).

```bash
panthera isolate \
    --tsv example/input/sample.tsv \
    --fasta genome/GRCh38.p14.genome.fasta \
    --gtf genome/gencode.v47.basic.annotation.gtf \
    --gene_target BRCA1 \
    --variant_target chr1-1000-A-T \
    --outdir example/output \
    --prefix ISOLATE_EXAMPLE 
```
You can find the variants in each combination under the column "block_variants" of the output file (`isolate_results.tsv`) in the output directory, alongside their respective delta scores.

The smallest combination of variants with the high delta scores are the likely causal variants.

<br />

## [PANTHERA API] How to run Panthera in a custom python script?
```python
from panthera.api import load_model, predict


# Specify DNA or RNA sequence in 5'-3' direction
seq = "GUAG"

# Load model
model = load_model("modelp")

# Predict
acceptor, donor = predict(seq, model)
```

The output is two numpy arrays: `acceptor` and `donor`. Each array contains the probabilities predicted for the corresponding base in the input sequence.

<br />

# FAQ
## 1. How can I speed up Panthera?

### Use GPU instead of CPU
Panthera uses deep learning models that execute much faster on GPUs. Make sure to use an environment configured with `tensorflow[and-cuda]` to use the GPU. Play with `--batch_size` to find the optimal number for your GPU memory.

Check if your GPU is recognized by TensorFlow:
```bash
python3 -c "import tensorflow as tf; print('GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"
```

### Use more CPUs
Even with a GPU, Panthera requires CPUs for pre- and post-processing steps (e.g., genetic background matching, sequence modifications, and delta score computations). Use the `--cpus` option to scale the number of parallel workers.

### Reduce computations
There are a few main ways to limit total computations:
1. When analyzing VCFs using `panthera survey`, if you know the population subgroup of your samples, specify `--genetic_background {AFR,AMR,EAS,EUR,SAS}` instead of analyzing all groups.
2. Reduce the context sequence distance. The default is `--context_dist 5000` (2500 bp upstream and downstream), but shorter values yield faster alignments if long-range effects are not a main concern.
3. Target specifically the genes of interest using the `--gene_target` parameter to bypass uninteresting transcript annotations.

<br />

## 2. Can I analyze non-human OR variants phased on non-GRCh38/hg38 reference genomes?
For SURVEY, Panthera currently only supports the analysis on GRCh38/hg38 as the background variants are only found in this reference genome. For ISOLATE, Panthera supports the analysis of variants on any reference genome as long as the user provides the variants and the corresponding reference genome.

<br />

# Troubleshooting
### JIT compilation failed
If `tf.config.optimizer.set_jit(True)` complains under GPU conditions, you may have incompatible CUDA drivers. Install a validated `tensorflow[and-cuda]` package for your CUDA version or run the pipeline on pure CPU (`CUDA_VISIBLE_DEVICES=""`).

### Variant Overlap errors
If the background VCF or user TSV specifies overlapping variant positions (e.g. massive INDELs that span adjacent SNVs), Panthera's interval resolution mechanism will attempt to drop the conflicting background alleles (with `--resolve_variant_conflicts`), or otherwise raise an error so silent data corruption does not occur.
