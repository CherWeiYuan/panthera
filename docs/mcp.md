## [PANTHERA MCP] How to run Panthera as a MCP server?

Panthera exposes its pipelines as [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) tools, 
allowing LLM-based clients (e.g. Claude Desktop, Cursor, Antigravity) to invoke 
`survey`, `isolate`, `query_fasta`, and `query_genomic_range` directly.

---

### Prerequisites
- Panthera cloned/installed locally (see [Installation](../README.md#installation)).
- The [genome folder](../README.md#required-files) downloaded and accessible on disk.

<br />

---

### Set up
1. Install Panthera locally (see [Installation](../README.md#installation)).
2. Download the [genome folder](../README.md#required-files) and make it accessible on disk.
3. Edit your MCP configuration file:

For example, in Antigravity, edit your `mcp_config.json` file with the following:
```json
{
  "mcpServers": {
    "panthera": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "panthera.core", "mcp"]
    }
  }
}
```

> **Replace** `/path/to/venv/bin/python` with the Python from your virtual environment.  
> Find it by running: `which python` (when the venv is activated) or `python -c "import sys; print(sys.executable)"`.

---
### Run Panthera MCP server
Just talk to the LLM normally.

```txt
Run panthera on this tsv:

chrom	pos	ref	alt
chr3	37007584	C	G

Tell me if there are any splice haplotypes detected.

My genetic background folder is downloaded to <path/to/genetic_background_vcf>.
```

#### LLM's (Gemini 3 Flash) response:

The Panthera survey for the variant chr3:37007584 C>G detected several spliceogenic haplotypes across different genetic backgrounds.

**Summary of Findings**

**Gene Affected:** MLH1

**Single Variant Impact:** In the reference (BASE) background, the variant alone has a masked delta score of 0.133, indicating it is likely spliceogenic.

**Haplotype Interactions:** In certain genetic backgrounds, the predicted spliceogenicity is significantly higher. Most notably, in individual NA19238 (Haplotype B), a haplotype containing this variant (along with chr3-37005942-G-A, chr3-37006220-C-T, and chr3-37007718-G-A) reached a masked delta score of 0.383.

**Spliceogenic Haplotypes (Masked Delta ≥ 0.1):**
The following are representative haplotypes where the variant was detected with significant spliceogenic potential:

| Individual | Haplotype | Variant Block | Masked Delta |
|------------|-----------|---------------|--------------|
| BASE | SINGLE_VARIANT | chr3-37007584-C-G | 0.133 |
| NA19238 | Haplotype B | chr3-37005942-G-A.chr3-37006220-C-T.chr3-37007584-C-G.chr3-37007718-G-A... | 0.383 |
| NA19983 | Haplotype A | chr3-37005228-C-T.chr3-37005232-G-A.chr3-37005942-G-A...chr3-37007584-C-G... | 0.125 |
| HG03683 | Haplotype A | chr3-37005942-G-A.chr3-37007584-C-G.chr3-37009387-G-A | 0.124 |


**Recommendations**
Since the variant is part of a complex haplotype in the most spliceogenic case (NA19238, delta 0.383), I recommend running Panthera Isolate to determine if chr3:37007584 is the primary driver or if other variants in that block are contributing to the effect.

Would you like me to run the isolation analysis on that specific haplotype?

```txt
Yes
```

#### LLM's (Gemini 3 Flash) response:

The isolation analysis for the haplotype block on MLH1 has identified a specific pair of variants that drive the high spliceogenic prediction.

Isolation Results
While the target variant chr3:37007584 C>G is spliceogenic on its own (delta 0.133), its effect is nearly tripled when combined with chr3:37007718 G>A.

| Variants in Combination | Masked Delta | Conclusion |
|-------|-----|-----| 
| chr3-37007584-C-G (Alone) | 0.133 | Spliceogenic baseline |
| chr3-37007584-C-G + chr3-37007718-G-A | 0.400 | Primary causal pair |
| chr3-37007584-C-G + chr3-37005942-G-A | 0.123 | No significant interaction |
| chr3-37007584-C-G + chr3-37006220-C-T | 0.117 | Slight decrease |
| chr3-37007584-C-G + chr3-37009658-... | 0.132 | No significant interaction |

**Key Insight**

The variant chr3:37007718 G>A acts as a strong modifier that enhances the spliceogenic impact of your target variant. When both are present, the masked delta score jumps to 0.400, which is a very strong prediction for altered splicing.

**Next Steps**

1. Minigene Assay: I strongly recommend performing a minigene assay in the wet lab to validate this predicted interaction.
2. IGV Visualization: If you'd like, I can generate a WIG file for you to visualize these splice site probabilities in IGV to see exactly where the splicing is being disrupted.
Would you like me to generate the visualization files?

---

### Available MCP tools

| Tool | Description |
|------|-------------|
| `survey` | Screen variants in a phased VCF or TSV for spliceogenic haplotypes |
| `isolate` | Identify the minimal causal variant set within a spliceogenic haplotype block |
| `query_fasta` | Predict splice-site probabilities for sequences in a FASTA file |
| `query_genomic_range` | Predict splice-site probabilities for a specific genomic region |

Each tool mirrors its CLI counterpart and accepts the same parameters. 
The `fasta` and `gtf` parameters must be **absolute paths** to the genome files 
downloaded in [Required files](#required-files).

---

### Troubleshooting

- **Server not found / command fails**: ensure the `cd` path is correct and that running `uv run python -m panthera.core mcp` manually in that directory succeeds.
- **GPU not used**: set the environment variable `CUDA_VISIBLE_DEVICES=0` inside the `args` string, e.g. `"CUDA_VISIBLE_DEVICES=0 uv run python -m panthera.core mcp"`.

<br />