## Genetic background data preparation workflow
One of Panthera's key feature is its SURVEY module which analyzes splice variants on different genetic background.
- To use this feature, Panthera needs fully phased VCFs, where each VCF represents all the variants in an individual's genome, including both haplotypes.
- Panthera is, by default, coded to use [up to 35 genetic backgrounds from 5 populations (African, Admixed American, East Asian, European and South Asian)](https://doi.org/10.1126%2Fscience.abf7117).

These processed phased VCFs can be downloaded from the [genome folder in Google Drive](https://drive.google.com/drive/folders/1-_7Tl3mVknu1TPKGLl-fIBCkflXFnLxm?usp=sharing) and placed in the path `src/panthera/data/genetic_background_vcf`, which in turn enables `panthera survey` calls (see `panthera survey -h` for more information).

Alternatively, you can process your fully phased VCFs using the following shell script workflow:

### 1. Define your environment parameters
```sh
dir="/path/to/your/working_directory" # Replace with your actual path
threads=4 # Replace with your desired number of threads
```

### 2. Download resolved haplotype VCFs
```sh
mkdir -p $dir/genome/reference_haplotypes

curl -o $dir/genome/reference_haplotypes/variants_freeze4_indel_insdel_alt.vcf.gz \
ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/HGSVC2/release/v2.0/integrated_callset/variants_freeze4_indel_insdel_alt.vcf.gz

curl -o $dir/genome/reference_haplotypes/variants_freeze4_snv_snv_alt.vcf.gz \
ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/HGSVC2/release/v2.0/integrated_callset/variants_freeze4_snv_snv_alt.vcf.gz
```

### 3. Index VCF
```sh
bcftools index $dir/genome/reference_haplotypes/variants_freeze4_snv_snv_alt.vcf.gz
bcftools index $dir/genome/reference_haplotypes/variants_freeze4_indel_insdel_alt.vcf.gz
```

### 4. Concatenate SNP and INDEL VCF
```sh
bcftools concat \
  -o $dir/genome/reference_haplotypes/variants_freeze4_snv_indel.vcf \
  --threads $threads \
  --allow-overlaps \
  $dir/genome/reference_haplotypes/variants_freeze4_snv_snv_alt.vcf.gz \
  $dir/genome/reference_haplotypes/variants_freeze4_indel_insdel_alt.vcf.gz

bgzip \
  -c $dir/genome/reference_haplotypes/variants_freeze4_snv_indel.vcf > \
  $dir/genome/reference_haplotypes/variants_freeze4_snv_indel.vcf.gz

bcftools index $dir/genome/reference_haplotypes/variants_freeze4_snv_indel.vcf.gz
```

### 5. Split VCF file by sample
```sh
mkdir -p $dir/genome/reference_haplotypes/haplotype_vcf

bcftools +split \
  $dir/genome/reference_haplotypes/variants_freeze4_snv_indel.vcf.gz \
  -Oz -o $dir/genome/reference_haplotypes/haplotype_vcf
```
