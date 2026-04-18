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

### Run from a pip-installed package

If you installed Panthera via `pip install panthera`, you can point directly to 
the Python executable in your virtual environment:

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