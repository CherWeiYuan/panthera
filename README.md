# Panthera
![Python Versions](https://img.shields.io/badge/python-3.10%20|%203.11-blue)
[![Python Tests](https://github.com/CherWeiYuan/panthera/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/CherWeiYuan/panthera/actions/workflows/test.yml)
![PyPI](https://img.shields.io/pypi/v/panthera-splice)
![License](https://img.shields.io/github/license/CherWeiYuan/panthera)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A toolkit for splice haplotype prediction and validation.
<br />

## Setting up Panthera

### Required files
1. GRCh38 genome fasta
2. GRCh38 GTF
3. Reference haplotype VCFs

<br />

All three files are found in the `genome` folder, which can be downloaded via the following options:

**Option 1**: Zenodo (install via `pip install zenodo-get`):
```bash
zenodo_get -v 4 21199785
tar -xvf genome.tar.xz
```

**Option 2**: gdown (install via `pip install gdown`):
```bash
gdown 1jCcQxtPTLDhuH7wBPsw056BfJoIEuy9I
tar -xvf genome.tar.xz
```

**Option 3**: Google Drive
Download the compressed folder directly from Google Drive [here](https://drive.google.com/drive/folders/1-_7Tl3mVknu1TPKGLl-fIBCkflXFnLxm?usp=sharing)

<br />

Checking file integrity via MD5 checksum
```
md5sum genome.tar.xz
```
The MD5 hash must be `bc68efaa37f96d8b91eeabdb74e1c178`. If not, re-download the `genome` folder.

<br />

### Installation
**IMPORTANT**: Panthera requires Python version 3.10 or **3.11 (recommended)**.

1. Set up an environment for Panthera (optional but recommended). 

Here is an example using mamba:
```bash
mamba create -n panthera python=3.11 pip
```

2. Install Panthera via pip or github clone:

Option 1: Pip
```bash
pip install --upgrade pip
pip install panthera-splice
```

Option 2: Clone the GitHub repository and run local installation:
```bash
git clone https://github.com/CherWeiYuan/panthera.git
cd panthera
pip install -e .
```

<br />

## SURVEY: Predicting Splice Haplotypes
Use the `survey` subcommand to predict splice haplotypes in a VCF or TSV.

### VCF input
You can run SURVEY using VCF obtained from variant calling (e.g., from DeepVariant):

**Step 1:** Run [WhatsHap](https://whatshap.readthedocs.io/en/latest/guide.html) to phase your VCF:
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

<br />

**Step 2:** Run Panthera SURVEY on the phased vcf:
```bash
panthera survey \
    --phased_vcf demo/input/demo_survey.vcf \
    --fasta genome/fasta/GRCh38.p14.genome.fasta \
    --gtf genome/gtf/gencode.v47.basic.annotation.gtf \
    --genetic_background_dir genome/reference_haplotypes \
    --outdir demo/output/survey_vcf \
    --prefix demo_survey_vcf
```

### TSV input
You can query variant(s) by manually creating a tab-separated file (TSV) with columns: `chrom`, `pos`, `ref`, `alt`:

```bash
panthera survey \
    --tsv demo/input/demo_survey.tsv \
    --fasta genome/fasta/GRCh38.p14.genome.fasta \
    --gtf genome/gtf/gencode.v47.basic.annotation.gtf \
    --genetic_background_dir genome/reference_haplotypes \
    --outdir demo/output/survey_tsv \
    --prefix demo_survey_tsv
```

For more options, see `panthera survey --help`.

<br />

## Isolate: Finding Causal Variants
A spliceogenic haplotype block consists of multiple variants but not all variants are drivers or modifiers. The `isolate` subcommand identifies the causal variants from your survey results or a manually created TSV (columns: `chrom`, `pos`, `ref`, `alt`).

To identify the causal variants, run Panthera ISOLATE on the tab-separated values (TSV) file of variants. The TSV can be obtained from Panthera SURVEY or created manually on a text file with 4 tab-separated columns: chrom, pos, ref, alt.

- Use the `-v` / `--variant_target` toggle to specify the target variant that must appear in every combination (format: `chrom-pos-ref-alt`).
- Use `--gene_target` to select the gene you want to target (useful when there are multiple genes in the same genomic region).

```bash
panthera isolate \
    --tsv demo/input/demo_isolate.tsv \
    --fasta genome/fasta/GRCh38.p14.genome.fasta \
    --gtf genome/gtf/gencode.v47.basic.annotation.gtf \
    --gene_target MLH1 \
    --variant_target chr3-37007584-C-G \
    --outdir demo/output/isolate \
    --prefix demo_isolate
```

You can find the variants in each combination under the column "block_variants" of the output file (`isolate_results.tsv`) in the output directory, alongside their respective delta scores.

The smallest combination of variants with the high delta scores are the likely causal variants.

<br />

### Output
Results are saved to your specified --outdir as `survey_results.tsv` or `isolate_results.tsv`.

The TSV contains the following columns:
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

<br />

## Python API
You can easily integrate Panthera into custom Python scripts for programmatic predictions and IGV visualizations.

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


You can plot the splice site probabilities in a wig file for visualization in IGV.
```python
from panthera.api import wig

# Create WIG file
wig(acceptor, 
    donor, 
    chrom = "chr1",    # Chromosome name; must match name in IGV.
    start = 1,         # 1-based
    strand = "+",      # Strand of the gene: "+" or "-"
    outdir = "./",     # Output directory
    prefix = "test"    # Prefix for the output files
)
```

<br />

## Natural Language (MCP) Integration
Panthera supports execution via natural language using LLMs (like Gemini). Refer to the [MCP guide](docs/mcp.md#mcp) to set up the MCP server.

**Example Prompt**:
```txt
Run panthera on this TSV: `chr3 37007629 A G` and `chr3 37007718 G A`. My genetic background folder is downloaded to genome/reference_haplotypes
```

The LLM will automatically format the inputs, execute the survey/isolate pipelines, and summarize the spliceogenicity, delta scores, and recommended next steps for you in plain English.

<br />

## Running Tests
To run the test suite, install the package along with its testing dependencies directly from the source repository:
```bash
git clone https://github.com/CherWeiYuan/panthera.git
cd panthera
pip install -e .
```

Run the tests via pytest:
```bash
pytest
```

A successful test result looks like this:
```sh
.............................................................................................................. [ 25%]
.............................................................................................................. [ 50%]
.............................................................................................................. [ 75%]
.............................................................................................................. [100%]
440 passed in 7.71s
```

<br />

## FAQ
### 1. How can I speed up Panthera?

**Limit Scope**: Reduce computations by specifying a target population subgroup (`--genetic_background {AFR,AMR,EAS,EUR,SAS}`), decreasing context sequence distance (`--context_dist`), or targeting only specific genes (`--gene_target`).

**Use more CPUs**: Use the `--cpus` flag to increase parallel workers for pre/post-processing tasks

**Enable GPU Support**: Ensure your environment has `tensorflow[and-cuda]` installed. Tune the `--batch_size` parameter to maximize your specific GPU memory usage.

Check if your GPU is recognized by TensorFlow:
```bash
python3 -c "import tensorflow as tf; print('GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"
```


<br />

### 2. Can I analyze non-human OR variants phased on non-GRCh38/hg38 reference genomes?
**Survey subcommand**: Currently restricted to GRCh38/hg38, as the background variants are mapped to this reference.
**Isolate subcommand**: Supports any reference genome, provided you supply the corresponding variants and FASTA file.

### 3. How do I prepare phased VCF files to add more genetic backgrounds?
Please refer to our [Data Preparation Workflow](docs/prepare_phased_vcf.md) for detailed instructions on adding diverse human genetic variations.

<br />

## Troubleshooting
### JIT compilation failed
If `tf.config.optimizer.set_jit(True)` complains under GPU conditions, you may have incompatible CUDA drivers. Install a validated `tensorflow[and-cuda]` package for your CUDA version or run the pipeline on pure CPU (`CUDA_VISIBLE_DEVICES=""`).

### Variant Overlap errors
If the background VCF or user TSV specifies overlapping variant positions (e.g. massive INDELs that span adjacent SNVs), Panthera's interval resolution mechanism will attempt to drop the conflicting background alleles (with `--resolve_variant_conflicts`), or otherwise raise an error so silent data corruption does not occur.
