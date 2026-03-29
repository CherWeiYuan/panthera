import pytest

# Assuming your module is named panthera.mutation
from panthera.core.bio.mutation import (
    snp_mutation,
    insertion_mutation,
    deletion_mutation,
    substitute_mutation,
)
from panthera.utils.exceptions import (
    AlleleLengthError,
    UnexpectedRefError,
    ZeroIndexError,
)

# A standard sequence for testing:
# 1-based: 1 2 3 4 5 6 7 8
# 0-based: 0 1 2 3 4 5 6 7
# Seq:     A T C G A T C G
BASE_SEQ = "ATCGATCG"

# --- 1. SHARED BOUNDARY TESTS ---


@pytest.mark.parametrize(
    "mut_func, ref, alt, extra_kwargs",
    [
        (snp_mutation, "A", "T", {}),
        (insertion_mutation, "A", "ATT", {"in_symbol": ">"}),
        (deletion_mutation, "A", "", {"del_symbol": "<"}),
        (substitute_mutation, "AT", "GC", {"in_symbol": ">", "del_symbol": "<"}),
    ],
)
def test_out_of_bounds_and_zero_index(mut_func, ref, alt, extra_kwargs):
    """Test that all mutation functions correctly reject invalid coordinates."""
    # Test Zero/Negative Index
    with pytest.raises(ZeroIndexError):
        mut_func(seq=BASE_SEQ, pos=0, ref=ref, alt=alt, **extra_kwargs)
    with pytest.raises(ZeroIndexError):
        mut_func(seq=BASE_SEQ, pos=-5, ref=ref, alt=alt, **extra_kwargs)

    # Test Out of Bounds Index
    with pytest.raises(IndexError):
        mut_func(seq=BASE_SEQ, pos=99, ref=ref, alt=alt, **extra_kwargs)


# --- 2. FUNCTION-SPECIFIC TESTS ---


class TestSNPMutation:
    def test_successful_snp(self):
        # Change pos 3 ('C') to 'G' -> "ATGGATCG"
        result = snp_mutation(BASE_SEQ, pos=3, ref="C", alt="G")
        assert result == "ATGGATCG"

    def test_unexpected_reference(self):
        # pos 3 is 'C', but we incorrectly specify 'A' as ref
        with pytest.raises(UnexpectedRefError):
            snp_mutation(BASE_SEQ, pos=3, ref="A", alt="G")


class TestInsertionMutation:
    def test_successful_insertion(self):
        # Insert at pos 2 ('T'). ref='T', alt='TTA'.
        # Expectation based on your logic: marker ">>", remainder "TA"
        # Result: A >> TA TCGATCG
        result = insertion_mutation(BASE_SEQ, pos=2, ref="T", alt="TTA", in_symbol=">")
        assert result == "AT>>TACGATCG"

    def test_invalid_ref_length(self):
        with pytest.raises(AlleleLengthError):
            insertion_mutation(BASE_SEQ, pos=2, ref="TC", alt="TTA", in_symbol=">")


class TestDeletionMutation:
    def test_successful_deletion(self):
        # Delete at pos 2 ('T'). ref='TC', alt='T'
        # Expectation: marker "<"
        # Result: AT < GATCG
        result = deletion_mutation(BASE_SEQ, pos=2, ref="TC", alt="T", del_symbol="<")
        assert result == "AT<GATCG"

    def test_empty_alt_deletion(self):
        # Total deletion of pos 2 ('T'). ref='T', alt=''
        # Result: A < CGATCG
        result = deletion_mutation(BASE_SEQ, pos=2, ref="T", alt="", del_symbol="<")
        assert result == "A<CGATCG"

    def test_invalid_ref_length(self):
        with pytest.raises(AlleleLengthError):
            deletion_mutation(
                BASE_SEQ, pos=2, ref="T", alt="TCCCC", del_symbol="<"
            )  # Expected length 1


class TestSubstituteMutation:
    def test_equal_length_substitution(self):
        # Substitute pos 2 ('TC'). ref='TC', alt='AG'
        result = substitute_mutation(
            BASE_SEQ, pos=2, ref="TC", alt="AG", in_symbol=">", del_symbol="<"
        )
        assert result == "AAGGATCG"

    def test_substitution_with_insertion_padding(self):
        # ref is shorter than alt. pos 2 ('TC'). ref='TC', alt='AGGG'
        # Expectation: marker ">>" + "AGGG"
        result = substitute_mutation(
            BASE_SEQ, pos=2, ref="TC", alt="AGGG", in_symbol=">", del_symbol="<"
        )
        assert result == "A>>AGGGGATCG"

    def test_substitution_with_deletion_padding(self):
        # ref is longer than alt. pos 2 ('TCG'). ref='TCG', alt='AG'
        # Expectation: "AG" + marker "<"
        result = substitute_mutation(
            BASE_SEQ, pos=2, ref="TCG", alt="AG", in_symbol=">", del_symbol="<"
        )
        assert result == "AAG<ATCG"

    def test_invalid_lengths(self):
        # Test ref length <= 1
        with pytest.raises(AlleleLengthError):
            substitute_mutation(
                BASE_SEQ, pos=2, ref="T", alt="AG", in_symbol=">", del_symbol="<"
            )
        # Test alt length <= 1
        with pytest.raises(AlleleLengthError):
            substitute_mutation(
                BASE_SEQ, pos=2, ref="TC", alt="A", in_symbol=">", del_symbol="<"
            )


# --- 3. ADDITIONAL FUNCTION-SPECIFIC TESTS ---


class TestSNP:
    @pytest.mark.parametrize(
        "seq, pos, ref, alt, expected",
        [
            ("AAAATAAAAA", 5, "T", "A", "AAAAAAAAAA"),  # test_snp_mutation_1
            ("GAAAAAAAAA", 1, "G", "A", "AAAAAAAAAA"),  # test_snp_mutation_2
            ("AAAAAAAAAC", 10, "C", "A", "AAAAAAAAAA"),  # test_snp_mutation_3
        ],
    )
    def test_simple_snp_mutations(self, seq, pos, ref, alt, expected):
        """Test basic SNP mutations."""
        assert snp_mutation(seq, pos, ref, alt) == expected

    def test_snp_sequential_mutation(self):
        """Test sequential SNP mutation."""
        result = snp_mutation("AAAAAAAAAC", 10, "C", "A")
        result = snp_mutation(result, 10, "A", "T")
        result = snp_mutation(result, 1, "A", "C")
        assert result == "CAAAAAAAAT"

    def test_snp_zero_index_error(self):
        """Test if zero-index error is detected."""
        with pytest.raises(ZeroIndexError):
            snp_mutation("AAAAAAAAAC", 0, "C", "A")

    def test_snp_ref_error(self):
        """Test if wrong reference allele is detected."""
        with pytest.raises(UnexpectedRefError):
            snp_mutation("AAAAAAAAAA", 1, "C", "A")


class TestInsertion:
    @pytest.mark.parametrize(
        "seq, pos, ref, alt, expected",
        [
            ("AAAAA", 5, "A", "ATTTTT", "AAAAA>>>>>TTTTT"),  # test_insertion_mutation_1
            ("AAAAA", 1, "A", "ATTTTT", "A>>>>>TTTTTAAAA"),  # test_insertion_mutation_2
            ("AAAAA", 3, "A", "ATTTTT", "AAA>>>>>TTTTTAA"),  # test_insertion_mutation_3
        ],
    )
    def test_simple_insertions(self, seq, pos, ref, alt, expected):
        """Test simple insertions."""
        assert insertion_mutation(seq, pos, ref, alt, in_symbol=">") == expected

    def test_insertion_sequential_mutation(self):
        """Test sequential insertion."""
        result = insertion_mutation("AAAAA", 1, "A", "ATTTTT", in_symbol=">")
        # result: A>>>>>TTTTTAAAA
        result = insertion_mutation(result, 15, "A", "AGG", in_symbol=">")
        # result: A>>>>>TTTTTAAAA>>GG
        result = insertion_mutation(result, 18, "G", "GCCAT", in_symbol=">")
        # result: A>>>>>TTTTTAAAA>>G>>>>CCATG
        result = insertion_mutation(result, 11, "T", "TATCG", in_symbol=">")
        # result: A>>>>>TTTTT>>>>ATCGAAAA>>G>>>>CCATG
        assert result == "A>>>>>TTTTT>>>>ATCGAAAA>>G>>>>CCATG"

    def test_insertion_zero_index_error(self):
        """Test if zero-index error is detected."""
        with pytest.raises(ZeroIndexError):
            insertion_mutation("AAAAAAAAAC", 0, "C", "CA", in_symbol=">")

    def test_insertion_ref_error(self):
        """Test if wrong reference allele is detected."""
        with pytest.raises(UnexpectedRefError):
            insertion_mutation("AAAAAAAAAC", 1, "C", "CA", in_symbol=">")


class TestDeletion:
    @pytest.mark.parametrize(
        "seq, pos, ref, alt, expected",
        [
            ("AAATT", 1, "AAA", "A", "A<<TT"),  # test_deletion_mutation_1
            ("AAAAT", 4, "AT", "A", "AAAA<"),  # test_deletion_mutation_2
            (
                "ATCGATCGATTGAC",
                4,
                "GATCG",
                "G",
                "ATCG<<<<ATTGAC",
            ),  # test_deletion_mutation_3
            (
                "ATCGATCGATTGAC",
                1,
                "ATCGATCGATTGAC",
                "A",
                "A<<<<<<<<<<<<<",
            ),  # test_deletion_mutation_4
        ],
    )
    def test_simple_deletions(self, seq, pos, ref, alt, expected):
        """Test simple deletions."""
        assert deletion_mutation(seq, pos, ref, alt, del_symbol="<") == expected

    def test_deletion_zero_index_error(self):
        """Test if zero-index error is detected."""
        with pytest.raises(ZeroIndexError):
            deletion_mutation(
                "ATCGATCGATTGAC", 0, "ATCGATCGATTGAC", "A", del_symbol="<"
            )

    def test_deletion_ref_error(self):
        """Test if wrong reference allele is detected."""
        with pytest.raises(UnexpectedRefError):
            deletion_mutation("ATCGATCGATTGAC", 5, "CGAT", "C", del_symbol="<")


class TestMixedMutation:
    def test_mixed_1(self):
        """Test SNP and insertion."""
        result = snp_mutation("AAATTTCCCGGG", 2, "A", "T")
        # result: ATATTTCCCGGG
        result = insertion_mutation(result, 4, "T", "TCGCG", in_symbol=">")
        # result: ATAT>>>>CGCGTTCCCGGG
        result = snp_mutation(result, 4, "T", "G")
        # result: ATAG>>>>CGCGTTCCCGGG
        result = insertion_mutation(result, 4, "G", "GTTT", in_symbol=">")
        # result: ATAG>>>TTT>>>>CGCGTTCCCGGG
        result = insertion_mutation(result, 19, "T", "TAATAT", in_symbol=">")
        # result: ATAG>>>TTT>>>>CGCGT>>>>>AATATTCCCGGG
        assert result == "ATAG>>>TTT>>>>CGCGT>>>>>AATATTCCCGGG"

    def test_mixed_2(self):
        """Test SNP and deletion."""
        result = deletion_mutation("CGATCGGATACTTACTTT", 1, "CGAT", "C", del_symbol="<")
        # result: C<<<CGGATACTTACTTT
        result = snp_mutation(result, 1, "C", "T")
        # result: T<<<CGGATACTTACTTT
        result = snp_mutation(result, 5, "C", "G")
        # result: T<<<GGGATACTTACTTT
        result = deletion_mutation(result, 5, "GGGATA", "G", del_symbol="<")
        # result: T<<<G<<<<<CTTACTTT
        assert result == "T<<<G<<<<<CTTACTTT"

    def test_mixed_3(self):
        """Test insertion and deletion."""
        result = deletion_mutation("GCCTATTGGTATACTN", 5, "ATTGG", "A", del_symbol="<")
        # result: GCCTA<<<<TATACTN
        result = deletion_mutation(result, 10, "TATA", "T", del_symbol="<")
        # result: GCCTA<<<<T<<<CTN
        result = insertion_mutation(result, 1, "G", "GTA", in_symbol=">")
        # result: G>>TACCTA<<<<T<<<CTN
        result = insertion_mutation(result, 18, "C", "CCC", in_symbol=">")
        # result: G>>TACCTA<<<<T<<<C>>CCTN
        result = deletion_mutation(result, 22, "CT", "T", del_symbol="<")
        # result: G>>TACCTA<<<<T<<<C>>CC<N
        assert result == "G>>TACCTA<<<<T<<<C>>CC<N"

    def test_mixed_4(self):
        """Test SNP, insertion and deletion."""
        result = snp_mutation("TAGGTATAGTTCCGAT", 4, "G", "C")
        # result: TAGCTATAGTTCCGAT
        result = insertion_mutation(result, 6, "A", "ACCC", in_symbol=">")
        # result: TAGCTA>>>CCCTAGTTCCGAT
        result = deletion_mutation(result, 10, "CCCTAG", "C", del_symbol="<")
        # result: TAGCTA>>>C<<<<<TTCCGAT
        result = snp_mutation(result, 20, "G", "C")
        # result: TAGCTA>>>C<<<<<TTCCCAT
        result = insertion_mutation(result, 21, "A", "ATAT", in_symbol=">")
        # result: TAGCTA>>>C<<<<<TTCCCA>>>TATT
        result = deletion_mutation(result, 27, "TT", "T", del_symbol="<")
        # result: TAGCTA>>>C<<<<<TTCCCA>>>TAT<
        assert result == "TAGCTA>>>C<<<<<TTCCCA>>>TAT<"


# --- 4. EDGE CASE TESTS ---


class TestLowercaseSequenceInput:
    """
    Tests that mutation functions handle lowercase sequences correctly.
    _convert_uppercase uppercases ref and alt, but actual_ref comes from seq.
    snp_mutation uses .upper() on both sides; insertion/deletion do NOT,
    so lowercase seq causes UnexpectedRefError in those functions.
    """

    def test_snp_handles_lowercase_seq(self):
        """snp_mutation should work with lowercase sequences"""
        mt_seq = snp_mutation(seq="atcgatcg", pos=3, ref="C", alt="G")
        assert mt_seq == "atGgatcg"

    def test_insertion_fails_on_lowercase_seq(self):
        """insertion_mutation raises UnexpectedRefError on lowercase seq (no .upper() on actual_ref)."""
        with pytest.raises(UnexpectedRefError):
            insertion_mutation(seq="atcgatcg", pos=2, ref="T", alt="TTA", in_symbol=">")

    def test_deletion_fails_on_lowercase_seq(self):
        """deletion_mutation raises UnexpectedRefError on lowercase seq (no .upper() on actual_ref)."""
        with pytest.raises(UnexpectedRefError):
            deletion_mutation(seq="atcgatcg", pos=2, ref="TC", alt="T", del_symbol="<")

    def test_substitute_fails_on_lowercase_seq(self):
        """substitute_mutation raises UnexpectedRefError on lowercase seq."""
        with pytest.raises(UnexpectedRefError):
            substitute_mutation(
                seq="atcgatcg", pos=2, ref="TC", alt="AG", in_symbol=">", del_symbol="<"
            )


class TestCustomPlaceholderSymbols:
    """Tests that non-default placeholder symbols work correctly."""

    def test_insertion_with_custom_symbol(self):
        result = insertion_mutation(
            seq="ATCGATCG", pos=2, ref="T", alt="TTA", in_symbol="}"
        )
        assert result == "AT}}TACGATCG"

    def test_deletion_with_custom_symbol(self):
        result = deletion_mutation(
            seq="ATCGATCG", pos=2, ref="TC", alt="T", del_symbol="{"
        )
        assert result == "AT{GATCG"

    def test_substitute_with_custom_symbols(self):
        result = substitute_mutation(
            seq="ATCGATCG", pos=2, ref="TC", alt="AGGG", in_symbol="}", del_symbol="{"
        )
        assert result == "A}}AGGGGATCG"


class TestBoundaryPositions:
    """Tests mutation at the very last position in a sequence."""

    def test_snp_at_last_position(self):
        result = snp_mutation(seq="ATCG", pos=4, ref="G", alt="A")
        assert result == "ATCA"

    def test_insertion_at_last_position(self):
        result = insertion_mutation(
            seq="ATCG", pos=4, ref="G", alt="GCC", in_symbol=">"
        )
        assert result == "ATCG>>CC"

    def test_deletion_at_last_position(self):
        """Deletion at last position with empty alt."""
        result = deletion_mutation(seq="ATCG", pos=4, ref="G", alt="", del_symbol="<")
        assert result == "ATC<"
