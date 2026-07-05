## Genetic background data preparation workflow
One of Panthera's key feature is its SURVEY module which analyzes splice variants on different genetic background.
- To use this feature, Panthera needs fully phased VCFs, where each VCF represents all the variants in an individual's genome, including both haplotypes.
- Panthera is, by default, coded to use [up to 35 genetic backgrounds from 5 populations (African, Admixed American, East Asian, European and South Asian)](https://doi.org/10.1126%2Fscience.abf7117).

These processed phased VCFs can be downloaded (instructions provided in [README.md](../README.md)) and placed in the path `src/panthera/data/genetic_background_vcf`, which in turn enables `panthera survey` calls (see `panthera survey -h` for more information).

Alternatively, you can process your fully phased VCFs using the following shell script. Copy the entire block, set the variables at the top, and run it in a single shell session. `set -euo pipefail` ensures the script aborts immediately if any step fails.

```sh
set -euo pipefail

# --- 1. Define your environment parameters ---
dir="~/Desktop/genome" # Replace with your actual path
threads=4                             # Replace with number of threads

ws="$dir/genome/workspace"

# --- 2. Download resolved haplotype VCFs ---
echo "--- Downloading resolved haplotype VCFs ---"
mkdir -p "$ws"
curl -o "$ws/variants_freeze4_indel_insdel_alt.vcf.gz" \
  ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/HGSVC2/release/v2.0/integrated_callset/variants_freeze4_indel_insdel_alt.vcf.gz
curl -o "$ws/variants_freeze4_snv_snv_alt.vcf.gz" \
  ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/HGSVC2/release/v2.0/integrated_callset/variants_freeze4_snv_snv_alt.vcf.gz

# Verify MD5 checksums of downloaded files
echo "--- Checking md5sum ---"
echo "18fab39a470a148e164f3050dfc88306  $ws/variants_freeze4_indel_insdel_alt.vcf.gz" | md5sum -c -
echo "51547c76f52925d99068afd7cade0a0d  $ws/variants_freeze4_snv_snv_alt.vcf.gz" | md5sum -c -

# --- 3. Index VCFs ---
echo "--- Indexing VCFs ---"
bcftools index "$ws/variants_freeze4_snv_snv_alt.vcf.gz"
bcftools index "$ws/variants_freeze4_indel_insdel_alt.vcf.gz"

# --- 4. Concatenate SNP and INDEL VCFs ---
echo "--- Concatenating VCFs ---"
bcftools concat \
  --threads "$threads" \
  --allow-overlaps \
  -Oz -o "$ws/variants_freeze4_snv_indel.vcf.gz" \
  "$ws/variants_freeze4_snv_snv_alt.vcf.gz" \
  "$ws/variants_freeze4_indel_insdel_alt.vcf.gz"
bcftools index "$ws/variants_freeze4_snv_indel.vcf.gz"

# --- 5. Clean up VCF headers ---
echo "--- Cleaning VCF headers ---"
# INFO ID tag REF_TRF is not assigned a number, which will result in
# cyvcf's htslib printing warnings. Rewrite the header to fix this.
bcftools view -h "$ws/variants_freeze4_snv_indel.vcf.gz" \
  | sed 's/##INFO=<ID=REF_TRF,Number=\.,Type=Flag,Description="Variant intersects a reference TRF region">/##INFO=<ID=REF_TRF,Number=0,Type=Flag,Description="Variant intersects a reference TRF region">/' \
  > "$ws/header.txt"
bcftools reheader \
  -h "$ws/header.txt" \
  "$ws/variants_freeze4_snv_indel.vcf.gz" \
  -o "$ws/variants_freeze4_snv_indel.clean.vcf.gz"
tabix -p vcf "$ws/variants_freeze4_snv_indel.clean.vcf.gz"

# --- 6. Split VCF file by sample ---
echo "--- Splitting VCF into ---"
mkdir -p "$dir/genome/reference_haplotypes"
bcftools +split \
  "$ws/variants_freeze4_snv_indel.clean.vcf.gz" \
  -Oz -o "$dir/genome/reference_haplotypes/"
```