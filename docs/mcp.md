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
Run panthera on this TSV

chrom	pos	ref	alt
chr3	37007629	A	G
chr3	37007718	G	A

My genetic background folder is downloaded to <path/to/genetic_background_vcf>.
```

LLM's (Gemini 3.1 Pro) response:
```
I have successfully run the Panthera survey pipeline on the provided variants.

Here is a summary of the actions taken:

Created input_variants.tsv in your workspace with the specified variants.
Ran the Panthera survey tool using the modelp splice-site model, generating the results in the panthera/output_survey/survey_results.tsv file.

Results
The survey identified that the haplotype block containing these variants in the MLH1 gene yields a masked delta score ranging from 0.39 to 0.459 across various genetic backgrounds (which is well above the spliceogenic threshold of 0.1). This strongly indicates potential spliceogenicity.

Recommended Next Steps
* Isolate the Causal Variant: Because these modifications exist within larger haplotype blocks (some containing additional modifier mutations), it is highly recommended to run the Panthera isolate tool next. This will help identify the minimal set of variants that are driving the predicted spliceogenicity. Let me know if you would like me to proceed with running the isolate pipeline!
* Wet-Lab Validation: For any variants predicted to be highly spliceogenic, performing a minigene assay in the wet lab is recommended to validate these computational predictions.
```

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