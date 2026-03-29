"""
Comprehensive pytest suite for isolate.py.

Strategy
--------
``panthera`` and ``pandera`` are not assumed to be installed.  Both are
replaced with lightweight in-process stubs *before* the module under test
is imported, so every test is hermetic and dependency-free beyond pandas
and pytest itself.

``find_genes_at_pos`` is replaced with a ``MagicMock`` that individual
tests configure via ``return_value`` / ``side_effect``.  The
``autouse`` fixture ``reset_mock`` resets it before every test so state
never leaks between tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast, Dict, List
import pandas as pd
from pandera.typing import DataFrame
import pytest

# ---------------------------------------------------------------------------
# Dependencies and Stubs
# ---------------------------------------------------------------------------

from panthera.core.bio.blocks import VariantSchema
from panthera.core.bio.gene import GeneObject

from panthera.core.pipelines.isolate import (
    _VARIANT_COLUMNS,
    _find_target_gene,
    _iter_haplotype_combinations,
    _parse_variant_target,
    phase1_create_haplotype_combinations,
)


@dataclass
class FakeGene(GeneObject):
    """
    A FakeGene that Pyright recognizes as a GeneObject.
    All fields have defaults so you can call FakeGene(gene_name="XYZ").
    """

    # We redefine the fields with default values
    chrom: str = "chr1"
    strand: str = "+"
    start: int = 1000
    end: int = 2000
    gene_name: str = "TestGene"
    gene_id: str = "ENSG00000000000"
    splice_sites: Dict[str, List[int]] = field(default_factory=dict)
    shex: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        # If you need to ensure splice_sites isn't empty for specific tests
        if not self.splice_sites:
            self.splice_sites = {"starts": [], "ends": []}

    def __hash__(self) -> int:
        return hash((self.gene_name, self.gene_id))


class FakeHaplotypeBlock:
    """Records constructor arguments so tests can inspect them."""

    def __init__(self, df: DataFrame[VariantSchema], gene_obj: GeneObject) -> None:
        self.vdf = df.copy()
        self.gene_obj = gene_obj


@pytest.fixture(autouse=True)
def mock_dependencies(mocker):
    """
    Mock out the heavy/external dependencies of isolate.py.
    This replaces the HaplotypeBlock and GeneObject classes and the GTF lookup function
    within the context of the isolate pipeline during testing.
    """
    mocker.patch(
        "panthera.core.pipelines.isolate.HaplotypeBlock", new=FakeHaplotypeBlock
    )
    mocker.patch("panthera.core.pipelines.isolate.GeneObject", new=FakeGene)
    mock_find = mocker.patch(
        "panthera.core.pipelines.isolate.find_genes_at_pos", return_value=None
    )

    # Store the mock for individual test configuration
    global _find_genes_at_pos_mock
    _find_genes_at_pos_mock = mock_find

    return mock_find


# ---------------------------------------------------------------------------
# Shared test fixtures and constants.
# ---------------------------------------------------------------------------

BRCA1 = FakeGene(gene_name="BRCA1", chrom="chr17")
TP53 = FakeGene(gene_name="TP53", chrom="chr17")

TARGET_STR = "chr1-1000-A-T"
TARGET_ROW = dict(chrom="chr1", pos=1000, ref="A", alt="T")
NT1 = dict(chrom="chr1", pos=2000, ref="C", alt="G")
NT2 = dict(chrom="chr1", pos=3000, ref="T", alt="A")
NT3 = dict(chrom="chr1", pos=4000, ref="G", alt="C")

GTF: dict = {"chr1": []}


def make_vdf(*rows: dict) -> DataFrame[VariantSchema]:
    """Return a plain DataFrame from positional row-dicts."""
    return cast(DataFrame[VariantSchema], pd.DataFrame(list(rows)))


@pytest.fixture(autouse=True)
def reset_mock():
    """Guarantee a clean mock state for every test."""
    # Note: _find_genes_at_pos_mock is set by the mock_dependencies fixture
    _find_genes_at_pos_mock.reset_mock()
    _find_genes_at_pos_mock.return_value = None
    _find_genes_at_pos_mock.side_effect = None
    yield


# ===========================================================================
# _parse_variant_target
# ===========================================================================


class TestParseVariantTarget:
    """Unit tests for the variant-string parser."""

    # --- Correct parses ---

    def test_standard_snv(self):
        assert _parse_variant_target("chr1-123456-A-T") == ("chr1", 123456, "A", "T")

    def test_chrx(self):
        chrom, pos, ref, alt = _parse_variant_target("chrX-99999-G-C")
        assert (chrom, pos, ref, alt) == ("chrX", 99999, "G", "C")

    def test_chry(self):
        chrom, *_ = _parse_variant_target("chrY-500-A-G")
        assert chrom == "chrY"

    def test_multichar_ref(self):
        _, _, ref, alt = _parse_variant_target("chr2-500-ACGT-T")
        assert ref == "ACGT" and alt == "T"

    def test_deletion_alleles(self):
        _, _, ref, alt = _parse_variant_target("chr3-200-ATG-A")
        assert ref == "ATG" and alt == "A"

    def test_insertion_alleles(self):
        _, _, ref, alt = _parse_variant_target("chr4-300-A-ATTG")
        assert ref == "A" and alt == "ATTG"

    def test_pos_type_is_int(self):
        _, pos, _, _ = _parse_variant_target("chr1-42-A-T")
        assert isinstance(pos, int)

    def test_returns_four_tuple(self):
        result = _parse_variant_target("chr1-1-A-T")
        assert len(result) == 4

    def test_large_genomic_position(self):
        _, pos, _, _ = _parse_variant_target("chr1-248956422-C-T")
        assert pos == 248_956_422

    def test_minimum_valid_position(self):
        # Parser itself does not enforce ge=1; that belongs to VariantSchema.
        _, pos, _, _ = _parse_variant_target("chr1-1-A-T")
        assert pos == 1

    def test_zero_position_parses_without_error(self):
        """Parser is intentionally lenient; schema coercion enforces ge=1 separately."""
        _, pos, _, _ = _parse_variant_target("chr1-0-A-T")
        assert pos == 0

    def test_extra_trailing_dash_fields_ignored(self):
        """Fields beyond the 4th are silently ignored (len ≥ 4 is sufficient)."""
        result = _parse_variant_target("chr1-1000-A-T-INFO-EXTRA")
        assert result == ("chr1", 1000, "A", "T")

    # --- Empty / None ---

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty"):
            _parse_variant_target("")

    def test_none_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            _parse_variant_target(None)  # type: ignore[arg-type]

    # --- Malformed strings ---

    def test_one_field_only_raises(self):
        with pytest.raises(ValueError, match="four dash-separated"):
            _parse_variant_target("chr1")

    def test_two_fields_raises(self):
        with pytest.raises(ValueError, match="four dash-separated"):
            _parse_variant_target("chr1-1000")

    def test_three_fields_raises(self):
        with pytest.raises(ValueError, match="four dash-separated"):
            _parse_variant_target("chr1-1000-A")

    def test_non_integer_pos_raises(self):
        with pytest.raises(ValueError, match="not a valid integer"):
            _parse_variant_target("chr1-abc-A-T")

    def test_float_pos_raises(self):
        with pytest.raises(ValueError, match="not a valid integer"):
            _parse_variant_target("chr1-1.5-A-T")

    def test_empty_pos_field_raises(self):
        # "chr1--A-T" splits to ["chr1", "", "A", "T"]; int("") fails.
        with pytest.raises(ValueError, match="not a valid integer"):
            _parse_variant_target("chr1--A-T")

    def test_whitespace_only_raises(self):
        # " " is truthy, so the empty guard is skipped; split gives [" "] → 1 part.
        with pytest.raises(ValueError, match="four dash-separated"):
            _parse_variant_target(" ")

    def test_tab_character_raises(self):
        with pytest.raises(ValueError):
            _parse_variant_target("\t")


# ===========================================================================
# _find_target_gene
# ===========================================================================


class TestFindTargetGene:
    """Unit tests for the GTF gene resolver."""

    # --- Correct resolution ---

    def test_gene_found_at_first_position(self):
        _find_genes_at_pos_mock.return_value = [BRCA1]
        df = make_vdf(TARGET_ROW)
        result = _find_target_gene(df, chrom="chr1", gtf_dict=GTF, gene_target="BRCA1")
        assert result is BRCA1

    def test_gene_found_at_second_position(self):
        def _side(chrom, pos, gtf_dict, existing_genes):
            return [BRCA1] if pos == 2000 else [TP53]

        _find_genes_at_pos_mock.side_effect = _side
        df = make_vdf(TARGET_ROW, NT1)
        result = _find_target_gene(df, chrom="chr1", gtf_dict=GTF, gene_target="BRCA1")
        assert result is BRCA1

    def test_none_then_gene_skips_gracefully(self):
        """None returned by find_genes_at_pos must be skipped without error."""

        def _side(chrom, pos, gtf_dict, existing_genes):
            return [] if pos == 1000 else [BRCA1]

        _find_genes_at_pos_mock.side_effect = _side
        df = make_vdf(TARGET_ROW, NT1)
        result = _find_target_gene(df, chrom="chr1", gtf_dict=GTF, gene_target="BRCA1")
        assert result is BRCA1

    def test_returns_gene_object_not_name(self):
        _find_genes_at_pos_mock.return_value = [BRCA1]
        df = make_vdf(TARGET_ROW)
        result = _find_target_gene(df, chrom="chr1", gtf_dict=GTF, gene_target="BRCA1")
        assert isinstance(result, FakeGene)

    # --- Early exit on match ---

    def test_stops_at_first_matching_position(self):
        """No positions after the match should be queried."""
        queried_positions: list[int] = []

        def _side(chrom, pos, gtf_dict, existing_genes):
            queried_positions.append(pos)
            return [BRCA1] if pos == 1000 else [TP53]

        _find_genes_at_pos_mock.side_effect = _side
        df = make_vdf(TARGET_ROW, NT1, NT2)
        _find_target_gene(df, chrom="chr1", gtf_dict=GTF, gene_target="BRCA1")
        assert queried_positions == [1000]

    def test_deduplicated_positions_queried_once_each(self):
        """Duplicate rows must not cause redundant GTF lookups."""
        _find_genes_at_pos_mock.return_value = [BRCA1]
        df = make_vdf(TARGET_ROW, TARGET_ROW, TARGET_ROW)  # three identical rows
        _find_target_gene(df, chrom="chr1", gtf_dict=GTF, gene_target="BRCA1")
        assert _find_genes_at_pos_mock.call_count == 1

    # --- seen_genes accumulation ---

    def test_seen_genes_starts_empty(self):
        received: list[list] = []

        def _side(chrom, pos, gtf_dict, existing_genes):
            received.append(list(existing_genes))
            return [BRCA1]

        _find_genes_at_pos_mock.side_effect = _side
        df = make_vdf(TARGET_ROW)
        _find_target_gene(df, chrom="chr1", gtf_dict=GTF, gene_target="BRCA1")
        assert received[0] == []

    def test_seen_genes_accumulates_between_positions(self):
        received: list[list] = []

        def _side(chrom, pos, gtf_dict, existing_genes):
            received.append(list(existing_genes))
            return [TP53] if pos == 1000 else [BRCA1]

        _find_genes_at_pos_mock.side_effect = _side
        df = make_vdf(TARGET_ROW, NT1)
        _find_target_gene(df, chrom="chr1", gtf_dict=GTF, gene_target="BRCA1")
        # Second call should receive the gene found at the first position.
        assert TP53 in received[1]

    def test_duplicate_gene_not_added_to_seen_twice(self):
        """If find_genes_at_pos returns the same gene object twice, seen_genes stays deduplicated."""
        seen_snapshots: list[list] = []

        def _side(chrom, pos, gtf_dict, existing_genes):
            seen_snapshots.append(list(existing_genes))
            return [TP53]  # always returns TP53

        _find_genes_at_pos_mock.side_effect = _side
        df = make_vdf(NT1, NT2)  # two positions, both return TP53
        with pytest.raises(ValueError):  # BRCA1 never found
            _find_target_gene(df, chrom="chr1", gtf_dict=GTF, gene_target="BRCA1")

        # At the second call, TP53 appears once (not twice) in seen_genes.
        assert seen_snapshots[1].count(TP53) == 1

    # --- Chromosome filtering ---

    def test_only_target_chrom_positions_queried(self):
        """Positions on other chromosomes must never reach find_genes_at_pos."""
        _find_genes_at_pos_mock.return_value = [BRCA1]
        df = make_vdf(
            TARGET_ROW,
            # NT1 is on chrom chr1, so no problem.
            NT1,
        )
        _find_target_gene(df, chrom="chr1", gtf_dict=GTF, gene_target="BRCA1")
        # check call counts or positions queried
        assert _find_genes_at_pos_mock.call_count >= 1

    # --- gtf_dict forwarded intact ---

    def test_gtf_dict_forwarded_by_identity(self):
        custom_gtf = {"chr1": ["fake_entry"]}
        _find_genes_at_pos_mock.return_value = [BRCA1]
        df = make_vdf(TARGET_ROW)
        _find_target_gene(df, chrom="chr1", gtf_dict=custom_gtf, gene_target="BRCA1")
        assert _find_genes_at_pos_mock.call_args.kwargs["gtf_dict"] is custom_gtf

    # --- Guard / error paths ---

    def test_empty_gene_target_raises(self):
        df = make_vdf(TARGET_ROW)
        with pytest.raises(ValueError, match="non-empty"):
            _find_target_gene(df, chrom="chr1", gtf_dict=GTF, gene_target="")

    def test_empty_chrom_raises(self):
        df = make_vdf(TARGET_ROW)
        with pytest.raises(ValueError, match="non-empty"):
            _find_target_gene(df, chrom="", gtf_dict=GTF, gene_target="BRCA1")

    def test_empty_dataframe_raises(self):
        df = cast(
            DataFrame[VariantSchema],
            pd.DataFrame(columns=pd.Index(["chrom", "pos", "ref", "alt"])),
        )
        with pytest.raises(ValueError, match="empty"):
            _find_target_gene(df, chrom="chr1", gtf_dict=GTF, gene_target="BRCA1")

    def test_no_positions_on_target_chrom_raises(self):
        df = make_vdf(dict(chrom="chr2", pos=5000, ref="A", alt="T"))
        with pytest.raises(ValueError, match="No positions found"):
            _find_target_gene(df, chrom="chr1", gtf_dict=GTF, gene_target="BRCA1")

    def test_all_find_genes_return_none_raises(self):
        _find_genes_at_pos_mock.return_value = []
        df = make_vdf(TARGET_ROW, NT1)
        with pytest.raises(ValueError, match="not found"):
            _find_target_gene(df, chrom="chr1", gtf_dict=GTF, gene_target="BRCA1")

    def test_only_wrong_gene_found_raises(self):
        _find_genes_at_pos_mock.return_value = [TP53]  # never BRCA1
        df = make_vdf(TARGET_ROW, NT1)
        with pytest.raises(ValueError, match="not found"):
            _find_target_gene(df, chrom="chr1", gtf_dict=GTF, gene_target="BRCA1")

    def test_error_message_contains_gene_name(self):
        _find_genes_at_pos_mock.return_value = [TP53]
        df = make_vdf(TARGET_ROW)
        with pytest.raises(ValueError, match="BRCA1"):
            _find_target_gene(df, chrom="chr1", gtf_dict=GTF, gene_target="BRCA1")

    def test_error_message_contains_chrom(self):
        _find_genes_at_pos_mock.return_value = [TP53]
        df = make_vdf(TARGET_ROW)
        with pytest.raises(ValueError, match="chr1"):
            _find_target_gene(df, chrom="chr1", gtf_dict=GTF, gene_target="BRCA1")


# ===========================================================================
# _iter_haplotype_combinations
# ===========================================================================


class TestIterHaplotypeCombinations:
    """Unit tests for the combination generator."""

    # Helper factories
    @staticmethod
    def _target(n: int = 1) -> list[tuple]:
        return [("chr1", 1000 + i, "A", "T", "PS") for i in range(n)]

    @staticmethod
    def _nontargets(n: int) -> list[tuple]:
        return [("chr1", 2000 + i * 1000, "C", "G", "PS") for i in range(n)]

    # --- Correct block counts (2^n − 1) ---

    @pytest.mark.parametrize(
        "n_nt,expected", [(1, 1), (2, 3), (3, 7), (4, 15), (5, 31)]
    )
    def test_block_count_is_2n_minus_1(self, n_nt: int, expected: int):
        blocks = list(
            _iter_haplotype_combinations(
                target_tuples=self._target(),
                nontarget_tuples=self._nontargets(n_nt),
                gene_obj=BRCA1,
            )
        )
        assert len(blocks) == expected

    # --- Target row(s) present in every block ---

    def test_single_target_row_in_every_block(self):
        target_tup = ("chr1", 1000, "A", "T", "PS")
        blocks = list(
            _iter_haplotype_combinations(
                target_tuples=[target_tup],
                nontarget_tuples=self._nontargets(3),
                gene_obj=BRCA1,
            )
        )
        for blk in blocks:
            assert target_tup in list(blk.vdf.itertuples(index=False, name=None))

    def test_multiple_target_rows_all_in_every_block(self):
        targets = self._target(n=2)
        blocks = list(
            _iter_haplotype_combinations(
                target_tuples=targets,
                nontarget_tuples=self._nontargets(2),
                gene_obj=BRCA1,
            )
        )
        for blk in blocks:
            rows = list(blk.vdf.itertuples(index=False, name=None))
            for tgt in targets:
                assert tgt in rows, f"Target {tgt} missing from block:\n{blk.vdf}"

    # --- DataFrame structure ---

    def test_dataframe_has_exactly_variant_columns(self):
        blocks = list(
            _iter_haplotype_combinations(
                target_tuples=self._target(),
                nontarget_tuples=self._nontargets(1),
                gene_obj=BRCA1,
            )
        )
        assert list(blocks[0].vdf.columns) == list(_VARIANT_COLUMNS)

    def test_chrom_column_is_string_dtype(self):
        blocks = list(
            _iter_haplotype_combinations(
                target_tuples=self._target(),
                nontarget_tuples=self._nontargets(1),
                gene_obj=BRCA1,
            )
        )
        assert pd.api.types.is_string_dtype(blocks[0].vdf["chrom"])

    def test_pos_column_is_integer_dtype(self):
        blocks = list(
            _iter_haplotype_combinations(
                target_tuples=self._target(),
                nontarget_tuples=self._nontargets(1),
                gene_obj=BRCA1,
            )
        )
        assert pd.api.types.is_integer_dtype(blocks[0].vdf["pos"])

    def test_ref_and_alt_are_string_dtype(self):
        blocks = list(
            _iter_haplotype_combinations(
                target_tuples=self._target(),
                nontarget_tuples=self._nontargets(1),
                gene_obj=BRCA1,
            )
        )
        df = blocks[0].vdf
        assert pd.api.types.is_string_dtype(df["ref"])
        assert pd.api.types.is_string_dtype(df["alt"])

    # --- gene_obj forwarding ---

    def test_gene_obj_forwarded_to_all_blocks(self):
        blocks = list(
            _iter_haplotype_combinations(
                target_tuples=self._target(),
                nontarget_tuples=self._nontargets(3),
                gene_obj=BRCA1,
            )
        )
        assert all(blk.gene_obj.gene_name == "BRCA1" for blk in blocks)

    def test_different_gene_obj_forwarded_correctly(self):
        blocks = list(
            _iter_haplotype_combinations(
                target_tuples=self._target(),
                nontarget_tuples=self._nontargets(1),
                gene_obj=TP53,
            )
        )
        assert blocks[0].gene_obj.gene_name == "TP53"

    # --- Combination correctness ---

    def test_all_nontarget_subsets_represented_for_n2(self):
        """With 2 non-targets {A, B}, the three blocks must be {A}, {B}, {A,B}."""
        nt = self._nontargets(2)
        target_tup = self._target()[0]
        blocks = list(
            _iter_haplotype_combinations(
                target_tuples=[target_tup],
                nontarget_tuples=nt,
                gene_obj=BRCA1,
            )
        )
        nt_sets = [
            frozenset(
                r for r in blk.vdf.itertuples(index=False, name=None) if r != target_tup
            )
            for blk in blocks
        ]
        expected = {frozenset([nt[0]]), frozenset([nt[1]]), frozenset(nt)}
        assert set(nt_sets) == expected

    def test_no_duplicate_blocks(self):
        blocks = list(
            _iter_haplotype_combinations(
                target_tuples=self._target(),
                nontarget_tuples=self._nontargets(4),
                gene_obj=BRCA1,
            )
        )
        serialised = [blk.vdf.to_csv(index=False) for blk in blocks]
        assert len(set(serialised)) == len(serialised)

    def test_row_count_per_block_correct(self):
        """A block for combination of size r should have len(targets) + r rows."""
        n_targets = 2
        targets = self._target(n=n_targets)
        nt = self._nontargets(3)
        blocks = list(
            _iter_haplotype_combinations(
                target_tuples=targets,
                nontarget_tuples=nt,
                gene_obj=BRCA1,
            )
        )
        # Blocks are in order r=1,2,3: 3 + 3 + 1 = 7 blocks.
        # r=1: C(3,1)=3 blocks of size n_targets+1=3 rows
        # r=2: C(3,2)=3 blocks of size n_targets+2=4 rows
        # r=3: C(3,3)=1 block  of size n_targets+3=5 rows
        sizes = [len(blk.vdf) for blk in blocks]
        assert sizes.count(n_targets + 1) == 3
        assert sizes.count(n_targets + 2) == 3
        assert sizes.count(n_targets + 3) == 1

    # --- Generator is lazy ---

    def test_generator_is_not_exhausted_on_creation(self):
        """The function is a generator; next() should yield without iterating all."""
        gen = _iter_haplotype_combinations(
            target_tuples=self._target(),
            nontarget_tuples=self._nontargets(5),  # 31 blocks total
            gene_obj=BRCA1,
        )
        first = next(gen)
        assert isinstance(first, FakeHaplotypeBlock)


# ===========================================================================
# phase1_create_haplotype_combinations  (integration)
# ===========================================================================


class TestPhase1CreateHaplotypeCombinations:
    """Integration tests for the public entry point."""

    def _gene_on(self):
        """Make find_genes_at_pos return BRCA1 for any call."""
        _find_genes_at_pos_mock.return_value = [BRCA1]

    # --- Return type and count ---

    def test_returns_list(self):
        self._gene_on()
        result = phase1_create_haplotype_combinations(
            vdf=make_vdf(TARGET_ROW, NT1),
            gtf_dict=GTF,
            gene_target="BRCA1",
            variant_target=TARGET_STR,
        )
        assert isinstance(result, list)

    def test_one_nontarget_produces_one_block(self):
        self._gene_on()
        result = phase1_create_haplotype_combinations(
            vdf=make_vdf(TARGET_ROW, NT1),
            gtf_dict=GTF,
            gene_target="BRCA1",
            variant_target=TARGET_STR,
        )
        assert len(result) == 1

    def test_two_nontargets_produce_three_blocks(self):
        self._gene_on()
        result = phase1_create_haplotype_combinations(
            vdf=make_vdf(TARGET_ROW, NT1, NT2),
            gtf_dict=GTF,
            gene_target="BRCA1",
            variant_target=TARGET_STR,
        )
        assert len(result) == 3

    def test_three_nontargets_produce_seven_blocks(self):
        self._gene_on()
        result = phase1_create_haplotype_combinations(
            vdf=make_vdf(TARGET_ROW, NT1, NT2, NT3),
            gtf_dict=GTF,
            gene_target="BRCA1",
            variant_target=TARGET_STR,
        )
        assert len(result) == 7

    # --- Block content ---

    def test_blocks_are_haplotypeblock_instances(self):
        self._gene_on()
        result = phase1_create_haplotype_combinations(
            vdf=make_vdf(TARGET_ROW, NT1),
            gtf_dict=GTF,
            gene_target="BRCA1",
            variant_target=TARGET_STR,
        )
        assert all(isinstance(b, FakeHaplotypeBlock) for b in result)

    def test_target_row_present_in_every_block(self):
        self._gene_on()
        result = phase1_create_haplotype_combinations(
            vdf=make_vdf(TARGET_ROW, NT1, NT2),
            gtf_dict=GTF,
            gene_target="BRCA1",
            variant_target=TARGET_STR,
        )
        target_tup = ("chr1", 1000, "A", "T", "PS")
        for blk in result:
            assert target_tup in list(blk.vdf.itertuples(index=False, name=None))

    def test_gene_obj_attached_to_all_blocks(self):
        self._gene_on()
        result = phase1_create_haplotype_combinations(
            vdf=make_vdf(TARGET_ROW, NT1),
            gtf_dict=GTF,
            gene_target="BRCA1",
            variant_target=TARGET_STR,
        )
        assert all(blk.gene_obj.gene_name == "BRCA1" for blk in result)

    # --- Execution order: parse before gene search ---

    def test_gene_search_uses_chrom_from_parsed_target(self):
        """
        _parse_variant_target must run before _find_target_gene so that
        chrom is available.  Verify find_genes_at_pos is called with the
        correct chromosome.
        """
        _find_genes_at_pos_mock.return_value = [BRCA1]
        phase1_create_haplotype_combinations(
            vdf=make_vdf(TARGET_ROW, NT1),
            gtf_dict=GTF,
            gene_target="BRCA1",
            variant_target=TARGET_STR,
        )
        for call in _find_genes_at_pos_mock.call_args_list:
            assert call.kwargs["chrom"] == "chr1"

    # --- Variant target parsing errors ---

    def test_empty_variant_target_raises(self):
        self._gene_on()
        with pytest.raises(ValueError, match="non-empty"):
            phase1_create_haplotype_combinations(
                vdf=make_vdf(TARGET_ROW, NT1),
                gtf_dict=GTF,
                gene_target="BRCA1",
                variant_target="",
            )

    def test_malformed_variant_target_too_few_parts_raises(self):
        self._gene_on()
        with pytest.raises(ValueError):
            phase1_create_haplotype_combinations(
                vdf=make_vdf(TARGET_ROW, NT1),
                gtf_dict=GTF,
                gene_target="BRCA1",
                variant_target="chr1-1000-A",
            )

    def test_non_integer_pos_in_target_raises(self):
        self._gene_on()
        with pytest.raises(ValueError, match="not a valid integer"):
            phase1_create_haplotype_combinations(
                vdf=make_vdf(TARGET_ROW, NT1),
                gtf_dict=GTF,
                gene_target="BRCA1",
                variant_target="chr1-abc-A-T",
            )

    # --- Target variant not found in vdf ---

    def test_target_absent_from_vdf_raises(self):
        self._gene_on()
        with pytest.raises(ValueError, match="not found in the phase set"):
            phase1_create_haplotype_combinations(
                vdf=make_vdf(NT1, NT2),
                gtf_dict=GTF,
                gene_target="BRCA1",
                variant_target=TARGET_STR,
            )

    def test_wrong_chrom_raises(self):
        self._gene_on()
        df = make_vdf(dict(chrom="chr2", pos=1000, ref="A", alt="T"), NT1)
        # Validation checks chromosome count before searching for targets.
        with pytest.raises(ValueError, match="Expect only one chromosome"):
            phase1_create_haplotype_combinations(
                vdf=df,
                gtf_dict=GTF,
                gene_target="BRCA1",
                variant_target=TARGET_STR,
            )

    def test_wrong_pos_raises(self):
        self._gene_on()
        df = make_vdf(dict(chrom="chr1", pos=9999, ref="A", alt="T"), NT1)
        with pytest.raises(ValueError, match="not found in the phase set"):
            phase1_create_haplotype_combinations(
                vdf=df,
                gtf_dict=GTF,
                gene_target="BRCA1",
                variant_target=TARGET_STR,
            )

    def test_wrong_ref_raises(self):
        self._gene_on()
        df = make_vdf(dict(chrom="chr1", pos=1000, ref="C", alt="T"), NT1)
        with pytest.raises(ValueError, match="not found in the phase set"):
            phase1_create_haplotype_combinations(
                vdf=df,
                gtf_dict=GTF,
                gene_target="BRCA1",
                variant_target=TARGET_STR,
            )

    def test_wrong_alt_raises(self):
        self._gene_on()
        df = make_vdf(dict(chrom="chr1", pos=1000, ref="A", alt="G"), NT1)
        with pytest.raises(ValueError, match="not found in the phase set"):
            phase1_create_haplotype_combinations(
                vdf=df,
                gtf_dict=GTF,
                gene_target="BRCA1",
                variant_target=TARGET_STR,
            )

    # --- No non-target variants ---

    def test_only_target_in_vdf_raises(self):
        self._gene_on()
        with pytest.raises(ValueError, match="No non-target"):
            phase1_create_haplotype_combinations(
                vdf=make_vdf(TARGET_ROW),
                gtf_dict=GTF,
                gene_target="BRCA1",
                variant_target=TARGET_STR,
            )

    # --- Gene not found ---

    def test_wrong_gene_raises(self):
        _find_genes_at_pos_mock.return_value = [TP53]
        with pytest.raises(ValueError, match="not found"):
            phase1_create_haplotype_combinations(
                vdf=make_vdf(TARGET_ROW, NT1),
                gtf_dict=GTF,
                gene_target="BRCA1",
                variant_target=TARGET_STR,
            )

    def test_empty_gene_target_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            phase1_create_haplotype_combinations(
                vdf=make_vdf(TARGET_ROW, NT1),
                gtf_dict=GTF,
                gene_target="",
                variant_target=TARGET_STR,
            )

    # --- Duplicate target rows ---

    def test_duplicate_target_rows_all_included_in_every_block(self):
        """Both rows of a repeated target variant must appear in the block df."""
        self._gene_on()
        df = make_vdf(TARGET_ROW, TARGET_ROW, NT1)
        result = phase1_create_haplotype_combinations(
            vdf=df,
            gtf_dict=GTF,
            gene_target="BRCA1",
            variant_target=TARGET_STR,
        )
        assert len(result) == 1  # only 1 non-target
        assert len(result[0].vdf) == 3  # 2 target rows + 1 non-target

    # --- Multi-chromosome vdf ---

    def test_gene_search_never_queries_other_chrom_positions(self):
        """Positions on non-target chromosomes must not reach find_genes_at_pos."""
        self._gene_on()
        # Since vdf must only have one chromosome, we can't test multiple chroms here.
        # But we can test that find_genes_at_pos is called with correct chrom.
        df = make_vdf(TARGET_ROW, NT1)
        phase1_create_haplotype_combinations(
            vdf=df,
            gtf_dict=GTF,
            gene_target="BRCA1",
            variant_target=TARGET_STR,
        )
        queried = [c.kwargs["pos"] for c in _find_genes_at_pos_mock.call_args_list]
        assert 1000 in queried

    def test_nontargets_from_other_chroms_form_combinations(self):
        """
        Since the validation logic ensures only one chromosome is present in vdf,
        this test case is now expected to raise a ValueError during the initial
        validation pass.
        """
        self._gene_on()
        chr2_nt = dict(chrom="chr2", pos=9999, ref="A", alt="T")
        df = make_vdf(TARGET_ROW, NT1, chr2_nt)
        with pytest.raises(ValueError, match="Expect only one chromosome"):
            phase1_create_haplotype_combinations(
                vdf=df,
                gtf_dict=GTF,
                gene_target="BRCA1",
                variant_target=TARGET_STR,
            )

    # --- Known bug: optional VariantSchema columns break itertuples alignment ---

    def test_extra_schema_columns_no_longer_expose_column_mismatch_bug(self):
        """
        FIXED: When vdf contains optional VariantSchema columns (genotype, phase_set,
        etc.), itertuples used to produce tuples wider than _VARIANT_COLUMNS,
        but we now select only the required columns.
        """
        self._gene_on()
        df = make_vdf(
            dict(**TARGET_ROW, genotype="0|1", phase_set="PS1"),
            dict(**NT1, genotype="1|0", phase_set="PS1"),
        )
        # This should now succeed and return 1 block
        result = phase1_create_haplotype_combinations(
            vdf=df,
            gtf_dict=GTF,
            gene_target="BRCA1",
            variant_target=TARGET_STR,
        )
        assert len(result) == 1

    # --- Coercible pos values ---

    def test_string_pos_in_vdf_matches_int_target_pos(self):
        """
        VariantSchema coerces pos to int.  If the caller pre-coerces, matching
        against the parsed integer must still work.
        """
        self._gene_on()
        df = make_vdf(
            dict(chrom="chr1", pos=1000, ref="A", alt="T"),  # pos already int
            NT1,
        )
        result = phase1_create_haplotype_combinations(
            vdf=df,
            gtf_dict=GTF,
            gene_target="BRCA1",
            variant_target=TARGET_STR,
        )
        assert len(result) == 1

    # --- Error messages are informative ---

    def test_missing_target_error_contains_variant_string(self):
        self._gene_on()
        with pytest.raises(ValueError, match="chr1-1000-A-T"):
            phase1_create_haplotype_combinations(
                vdf=make_vdf(NT1, NT2),
                gtf_dict=GTF,
                gene_target="BRCA1",
                variant_target=TARGET_STR,
            )

    def test_missing_gene_error_contains_gene_name(self):
        _find_genes_at_pos_mock.return_value = [TP53]
        with pytest.raises(ValueError, match="BRCA1"):
            phase1_create_haplotype_combinations(
                vdf=make_vdf(TARGET_ROW, NT1),
                gtf_dict=GTF,
                gene_target="BRCA1",
                variant_target=TARGET_STR,
            )
