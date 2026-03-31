import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, cast

import pandas as pd
from tqdm import tqdm

# Configure module-level logger
logger = logging.getLogger(__name__)


@dataclass
class GeneObject:
    """Represents a genomic Gene with its coordinates and splice sites."""

    chrom: str
    strand: str
    start: int
    end: int
    gene_name: str
    gene_id: str
    splice_sites: Dict[str, List[int]]
    shex: List[List[int]]


class GTFParser:
    """Class to handle parsing, processing, and caching of GTF files."""

    SHALLOW_INTRON_OFFSET = 149
    WEAK_TRANSCRIPT_LEVEL = "5"

    def __init__(self, gtf_file: str):
        self.gtf_file = Path(gtf_file)
        self.json_cache = self.gtf_file.with_suffix(".json")
        self._gtf_df: Optional[pd.DataFrame] = None

    def _load_gtf_to_dataframe(self) -> pd.DataFrame:
        """Loads and parses the GTF file into a Pandas DataFrame."""
        if self._gtf_df is not None:
            return self._gtf_df

        logger.info(f"Loading GTF file: {self.gtf_file}")

        # If Pandas sees '.gz', it will decompress the file automatically
        df = pd.read_csv(
            self.gtf_file,
            sep="\t",
            comment="#",
            header=0,
            names=[
                "seqname",
                "source",
                "feature",
                "start",
                "end",
                "score",
                "strand",
                "frame",
                "attribute",
            ],
            dtype={
                "seqname": str,
                "source": str,
                "feature": str,
                "start": int,
                "end": int,
                "score": str,
                "strand": str,
                "frame": str,
                "attribute": str,
            },
        )

        # Standardize chromosome names
        df["seqname"] = df["seqname"].apply(
            lambda x: x if x.startswith("chr") else f"chr{x}"
        )

        # Parse attributes
        logger.info("Parsing attributes with vectorized regex...")
        attributes_df = self._parse_attributes(cast(pd.Series, df["attribute"]))

        # Combine with the main dataframe and drop the raw attribute column
        self._gtf_df = pd.concat(
            [df.drop(columns=["attribute"]), attributes_df], axis=1
        )

        return self._gtf_df

    @staticmethod
    def _parse_attributes(attribute_series: pd.Series) -> pd.DataFrame:
        """Parses GTF attribute strings using vectorized regex with
        robust whitespace and word boundary handling.
        """
        target_keys = [
            "gene_id",
            "gene_name",
            "transcript_id",
            "exon_number",
            "transcript_support_level",
        ]

        extracted_data = {}

        for key in target_keys:
            # \b ensures we only match the exact key, not substrings like 'my_gene_id'
            # \s+ allows for one or more spaces (or tabs) between the key and value
            pattern = rf'\b{key}\s+"([^"]+)"'
            extracted_data[key] = attribute_series.str.extract(pattern, expand=False)

        return pd.DataFrame(extracted_data)

    def get_gene_sites(self) -> Dict[str, Dict[str, Any]]:
        """Calculates acceptor, donor, and shallow intron/exon sites per gene.
        (Optimized via Pandas GroupBy)
        """
        df = self._load_gtf_to_dataframe()
        gene_site_dict = {}

        # 1. Pre-filter and pre-compute necessary metadata
        # Get gene_name and strand mappings to avoid doing this in the loop
        gene_meta = (
            cast(pd.DataFrame, df[["gene_id", "gene_name", "strand"]])
            .dropna(subset=["gene_id"])
            .drop_duplicates(subset=["gene_id"])
        )
        meta_dict = gene_meta.set_index("gene_id").to_dict("index")

        exons_df = df[df["feature"] == "exon"].copy()
        exons_df["exon_number"] = pd.to_numeric(
            exons_df["exon_number"], errors="coerce"
        )

        # Identify valid transcripts once globally
        valid_tx_mask = (df["feature"] == "transcript") & (
            df["transcript_support_level"] != self.WEAK_TRANSCRIPT_LEVEL
        )
        valid_transcript_ids = set(
            cast(pd.Series, df[valid_tx_mask]["transcript_id"]).dropna()
        )

        # 2. Group dataframe operations (O(1) lookups instead of O(N) masking)
        exons_by_gene = exons_df.groupby("gene_id")

        valid_exons = cast(
            pd.DataFrame,
            exons_df[
                cast(pd.Series, exons_df["transcript_id"]).isin(valid_transcript_ids)
            ],
        )
        exons_by_transcript = valid_exons.groupby("transcript_id")
        transcripts_per_gene = cast(
            pd.Series, valid_exons.groupby("gene_id")["transcript_id"].unique()
        )

        gene_id_list = df["gene_id"].dropna().unique()
        pbar = tqdm(
            total=len(gene_id_list),
            desc="Extracting splice sites " + "(performed only once per GTF file)",
        )

        for gene_id in gene_id_list:
            meta = meta_dict.get(gene_id)
            if not meta:
                pbar.update(1)
                continue

            gene_name = meta["gene_name"]
            gene_strand = meta["strand"]

            if gene_name in gene_site_dict.get(gene_id, {}):
                raise ValueError(
                    f"Data corruption: Gene {gene_name} duplicated in GTF processing."
                )

            # Initialize dict entry using sets for faster unique additions
            gene_site_dict.setdefault(gene_id, {})[gene_name] = {
                "acc": set(),
                "dnr": set(),
                "shex": [],
            }

            acc_set = gene_site_dict[gene_id][gene_name]["acc"]
            dnr_set = gene_site_dict[gene_id][gene_name]["dnr"]

            # Process SHEX
            if gene_id in exons_by_gene.groups:
                gene_exons = cast(
                    pd.DataFrame, exons_by_gene.get_group(gene_id)[["start", "end"]]
                ).drop_duplicates()
                for row in gene_exons.itertuples(index=False):
                    start, end = min(row[0], row[1]), max(row[0], row[1])
                    gene_site_dict[gene_id][gene_name]["shex"].append(
                        [
                            int(start - self.SHALLOW_INTRON_OFFSET),
                            int(end + self.SHALLOW_INTRON_OFFSET),
                        ]
                    )

            # Process ACC/DNR
            if gene_id in transcripts_per_gene.index:
                for t_id in transcripts_per_gene[gene_id]:
                    tr_exons = exons_by_transcript.get_group(t_id)
                    exon_nums = (
                        cast(pd.Series, tr_exons["exon_number"]).dropna().tolist()
                    )

                    if len(exon_nums) <= 1:
                        continue

                    first_exon, last_exon = min(exon_nums), max(exon_nums)

                    for row in cast(
                        pd.DataFrame, tr_exons[["start", "end", "exon_number"]]
                    ).itertuples(index=False):
                        start, end = (
                            int(min(row[0], row[1])),
                            int(max(row[0], row[1])),
                        )
                        exon_num = row[2]

                        if gene_strand == "+":
                            if exon_num == first_exon:
                                dnr_set.add(end)
                            elif exon_num == last_exon:
                                acc_set.add(start)
                            else:
                                acc_set.add(start)
                                dnr_set.add(end)
                        elif gene_strand == "-":
                            if exon_num == first_exon:
                                dnr_set.add(start)
                            elif exon_num == last_exon:
                                acc_set.add(end)
                            else:
                                acc_set.add(end)
                                dnr_set.add(start)

            # Convert sets to sorted lists at the end
            gene_site_dict[gene_id][gene_name]["acc"] = sorted(list(acc_set))
            gene_site_dict[gene_id][gene_name]["dnr"] = sorted(list(dnr_set))

            pbar.update(1)

        pbar.close()
        return gene_site_dict

    def get_gtf_dict(self) -> Dict[str, List[Any]]:
        """Generates or loads a cached dictionary mapping chromosomes to gene metadata."""
        if self.json_cache.exists():
            logger.info("GTF JSON cache found. Loading...")
            with open(self.json_cache, "r") as f:
                return json.load(f)

        logger.info("GTF JSON cache does not exist. Creating...")
        gene_site_dict = self.get_gene_sites()
        df = self._load_gtf_to_dataframe()
        gene_df = cast(pd.DataFrame, df[df["feature"] == "gene"])

        gtf_dict = {}

        for row in gene_df.itertuples():
            start, end = min(row.start, row.end), max(row.start, row.end)  # type: ignore

            # Using defaults gracefully if keys are missing
            gene_id = getattr(row, "gene_id", "")
            gene_name = getattr(row, "gene_name", "")

            site_data = gene_site_dict.get(gene_id, {}).get(gene_name, {})
            splice_sites = {
                "acc": site_data.get("acc", []),
                "dnr": site_data.get("dnr", []),
            }
            shex = site_data.get("shex", [])

            new_entry = [
                row.Index,  # type: ignore
                row.seqname,  # type: ignore
                start,
                end,
                row.strand,  # type: ignore
                gene_name,
                gene_id,
                splice_sites,
                shex,
            ]

            gtf_dict.setdefault(row.seqname, []).append(new_entry)  # type: ignore

        logger.info("Exporting GTF dictionary to JSON cache.")
        with open(self.json_cache, "w") as f:
            json.dump(gtf_dict, f)

        return gtf_dict


def find_genes_at_pos(
    chrom: str,
    pos: int,
    gtf_dict: Dict[str, List[Any]],
    existing_genes: List[GeneObject],
) -> List[GeneObject]:
    """Finds and returns GeneObjects found in a
    specific chromosome and genomic coordinate.
    """
    out = []
    obtained_gene_names = {gene.gene_name for gene in existing_genes}

    for val in gtf_dict.get(chrom, []):
        gene_name = str(val[5]).replace("'", "").replace('"', "")

        if gene_name in obtained_gene_names:
            continue

        start, end = int(val[2]), int(val[3])

        if start <= pos <= end:
            gene = GeneObject(
                chrom=str(val[1]),
                strand=str(val[4]),
                start=start,
                end=end,
                gene_name=gene_name,
                gene_id=str(val[6]).replace("'", "").replace('"', ""),
                splice_sites=val[7],
                shex=val[8],
            )
            out.append(gene)

    return out
