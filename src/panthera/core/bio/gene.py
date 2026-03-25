import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

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
    name: str
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
        attributes_df = pd.DataFrame(
            df["attribute"].apply(self._parse_attributes).tolist()
        )
        self._gtf_df = pd.concat(
            [df.drop(columns=["attribute"]), attributes_df], axis=1
        )

        return self._gtf_df

    @staticmethod
    def _parse_attributes(attr_string: str) -> Dict[str, str]:
        """Parses the GTF attribute string into a dictionary."""
        attributes = {}
        target_keys = {
            "gene_id",
            "gene_name",
            "transcript_id",
            "exon_number",
            "transcript_support_level",
        }

        for attribute in attr_string.strip().split(";"):
            if not attribute:
                continue
            parts = attribute.strip().split(" ", 1)
            if len(parts) == 2:
                key, value = parts
                if key in target_keys:
                    attributes[key] = value.strip('"')
        return attributes

    def get_gene_sites(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculates acceptor, donor, and shallow intron/exon sites per gene.
        """
        df = self._load_gtf_to_dataframe()
        gene_site_dict = {}
        gene_id_list = df["gene_id"].dropna().unique()

        pbar = tqdm(total=len(gene_id_list), desc="Extracting splice sites")

        for gene_id in gene_id_list:
            gene_df = df[df["gene_id"] == gene_id].reset_index(drop=True)
            if gene_df.empty:
                continue

            gene_name = gene_df.at[0, "gene_name"]
            gene_strand = gene_df.at[0, "strand"]

            # Initialize gene dictionary entry
            if gene_name in gene_site_dict.get(gene_id, {}):
                raise ValueError(
                    f"Data corruption: Gene {gene_name} "
                    + "duplicated in GTF processing."
                )

            # Calculate Shallow Intron + Exon (shex) positions
            shex = []
            exon_df = gene_df[gene_df["feature"] == "exon"].drop_duplicates(
                subset=["seqname", "start", "end", "strand"]
            )

            # Using itertuples for performance (much faster than iterrows)
            for row in exon_df.itertuples(index=False):
                start, end = min(row.start, row.end), max(row.start, row.end)
                shex.append(
                    [
                        int(start - self.SHALLOW_INTRON_OFFSET),
                        int(end + self.SHALLOW_INTRON_OFFSET),
                    ]
                )

            gene_site_dict.setdefault(gene_id, {})[gene_name] = {
                "acc": [],
                "dnr": [],
                "shex": shex,
            }

            # Filter transcripts
            valid_transcripts = gene_df[
                (gene_df["feature"] == "transcript")
                & (gene_df["transcript_support_level"] != self.WEAK_TRANSCRIPT_LEVEL)
            ]["transcript_id"].unique()

            for transcript_id in valid_transcripts:
                tr_exons = gene_df[
                    (gene_df["transcript_id"] == transcript_id)
                    & (gene_df["feature"] == "exon")
                ]
                exon_num_list = [int(i) for i in tr_exons["exon_number"].dropna() if i]

                if len(exon_num_list) <= 1:
                    logger.debug(f"Skipping {transcript_id}: Only one exon.")
                    continue

                first_exon, last_exon = min(exon_num_list), max(exon_num_list)

                for row in tr_exons.itertuples(index=False):
                    start = int(min(row.start, row.end))
                    end = int(max(row.start, row.end))
                    exon_num = int(row.exon_number)

                    acc_list = gene_site_dict[gene_id][gene_name]["acc"]
                    dnr_list = gene_site_dict[gene_id][gene_name]["dnr"]

                    if gene_strand == "+":
                        if exon_num == first_exon:
                            dnr_list.append(end)
                        elif exon_num == last_exon:
                            acc_list.append(start)
                        else:
                            acc_list.append(start)
                            dnr_list.append(end)
                    elif gene_strand == "-":
                        if exon_num == first_exon:
                            dnr_list.append(start)
                        elif exon_num == last_exon:
                            acc_list.append(end)
                        else:
                            acc_list.append(end)
                            dnr_list.append(start)
                    else:
                        raise ValueError(
                            f"Unspecified strand '{gene_strand}' for "
                            + f"{gene_name} {transcript_id}"
                        )

            # Deduplicate and sort sites
            gene_site_dict[gene_id][gene_name]["acc"] = sorted(
                list(set(gene_site_dict[gene_id][gene_name]["acc"]))
            )
            gene_site_dict[gene_id][gene_name]["dnr"] = sorted(
                list(set(gene_site_dict[gene_id][gene_name]["dnr"]))
            )

            pbar.update(1)

        pbar.close()
        return gene_site_dict

    def get_gtf_dict(self) -> Dict[str, List[Any]]:
        """
        Generates or loads a cached dictionary mapping chromosomes to gene metadata.
        """
        if self.json_cache.exists():
            logger.info("GTF JSON cache found. Loading...")
            with open(self.json_cache, "r") as f:
                return json.load(f)

        logger.info("GTF JSON cache does not exist. Creating...")
        gene_site_dict = self.get_gene_sites()
        df = self._load_gtf_to_dataframe()
        gene_df = df[df["feature"] == "gene"]

        gtf_dict = {}

        for row in gene_df.itertuples():
            start, end = min(row.start, row.end), max(row.start, row.end)

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
                row.Index,
                row.seqname,
                start,
                end,
                row.strand,
                gene_name,
                gene_id,
                splice_sites,
                shex,
            ]

            gtf_dict.setdefault(row.seqname, []).append(new_entry)

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
    """
    Finds and returns GeneObjects found in a specific chromosome and genomic coordinate.
    """
    out = []
    obtained_gene_names = {gene.name for gene in existing_genes}

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
                name=gene_name,
                gene_id=str(val[6]).replace("'", "").replace('"', ""),
                splice_sites=val[7],
                shex=val[8],
            )
            out.append(gene)

    return out
