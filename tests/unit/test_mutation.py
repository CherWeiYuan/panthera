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
    "mut_func, ref, alt",
    [
        (snp_mutation, "A", "T"),
        (insertion_mutation, "A", "ATT"),
        (deletion_mutation, "A", ""),
        (substitute_mutation, "AT", "GC"),
    ],
)
def test_out_of_bounds_and_zero_index(mut_func, ref, alt):
    """Test that all mutation functions correctly reject invalid coordinates."""
    # Test Zero/Negative Index
    with pytest.raises(ZeroIndexError):
        mut_func(seq=BASE_SEQ, pos=0, ref=ref, alt=alt)
    with pytest.raises(ZeroIndexError):
        mut_func(seq=BASE_SEQ, pos=-5, ref=ref, alt=alt)

    # Test Out of Bounds Index
    with pytest.raises(IndexError):
        mut_func(seq=BASE_SEQ, pos=99, ref=ref, alt=alt)


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
        result = insertion_mutation(BASE_SEQ, pos=2, ref="T", alt="TTA")
        assert result == "AT>>TACGATCG"

    def test_invalid_ref_length(self):
        with pytest.raises(AlleleLengthError):
            insertion_mutation(BASE_SEQ, pos=2, ref="TC", alt="TTA")


class TestDeletionMutation:
    def test_successful_deletion(self):
        # Delete at pos 2 ('T'). ref='TC', alt='T'
        # Expectation: marker "<"
        # Result: AT < GATCG
        result = deletion_mutation(BASE_SEQ, pos=2, ref="TC", alt="T")
        assert result == "AT<GATCG"

    def test_empty_alt_deletion(self):
        # Total deletion of pos 2 ('T'). ref='T', alt=''
        # Result: A < CGATCG
        result = deletion_mutation(BASE_SEQ, pos=2, ref="T", alt="")
        assert result == "A<CGATCG"

    def test_invalid_ref_length(self):
        with pytest.raises(AlleleLengthError):
            deletion_mutation(
                BASE_SEQ, pos=2, ref="T", alt="TCCCC"
            )  # Expected length 1


class TestSubstituteMutation:
    def test_equal_length_substitution(self):
        # Substitute pos 2 ('TC'). ref='TC', alt='AG'
        result = substitute_mutation(BASE_SEQ, pos=2, ref="TC", alt="AG")
        assert result == "AAGGATCG"

    def test_substitution_with_insertion_padding(self):
        # ref is shorter than alt. pos 2 ('TC'). ref='TC', alt='AGGG'
        # Expectation: marker ">>" + "AGGG"
        result = substitute_mutation(BASE_SEQ, pos=2, ref="TC", alt="AGGG")
        assert result == "A>>AGGGGATCG"

    def test_substitution_with_deletion_padding(self):
        # ref is longer than alt. pos 2 ('TCG'). ref='TCG', alt='AG'
        # Expectation: "AG" + marker "<"
        result = substitute_mutation(BASE_SEQ, pos=2, ref="TCG", alt="AG")
        assert result == "AAG<ATCG"

    def test_invalid_lengths(self):
        # Test ref length <= 1
        with pytest.raises(AlleleLengthError):
            substitute_mutation(BASE_SEQ, pos=2, ref="T", alt="AG")
        # Test alt length <= 1
        with pytest.raises(AlleleLengthError):
            substitute_mutation(BASE_SEQ, pos=2, ref="TC", alt="A")


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
        assert insertion_mutation(seq, pos, ref, alt) == expected

    def test_insertion_sequential_mutation(self):
        """Test sequential insertion."""
        result = insertion_mutation("AAAAA", 1, "A", "ATTTTT")
        # result: A>>>>>TTTTTAAAA
        result = insertion_mutation(result, 15, "A", "AGG")
        # result: A>>>>>TTTTTAAAA>>GG
        result = insertion_mutation(result, 18, "G", "GCCAT")
        # result: A>>>>>TTTTTAAAA>>G>>>>CCATG
        result = insertion_mutation(result, 11, "T", "TATCG")
        # result: A>>>>>TTTTT>>>>ATCGAAAA>>G>>>>CCATG
        assert result == "A>>>>>TTTTT>>>>ATCGAAAA>>G>>>>CCATG"

    def test_insertion_zero_index_error(self):
        """Test if zero-index error is detected."""
        with pytest.raises(ZeroIndexError):
            insertion_mutation("AAAAAAAAAC", 0, "C", "CA")

    def test_insertion_ref_error(self):
        """Test if wrong reference allele is detected."""
        with pytest.raises(UnexpectedRefError):
            insertion_mutation("AAAAAAAAAC", 1, "C", "CA")


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
        assert deletion_mutation(seq, pos, ref, alt) == expected

    def test_deletion_zero_index_error(self):
        """Test if zero-index error is detected."""
        with pytest.raises(ZeroIndexError):
            deletion_mutation("ATCGATCGATTGAC", 0, "ATCGATCGATTGAC", "A")

    def test_deletion_ref_error(self):
        """Test if wrong reference allele is detected."""
        with pytest.raises(UnexpectedRefError):
            deletion_mutation("ATCGATCGATTGAC", 5, "CGAT", "C")


class TestMixedMutation:
    def test_mixed_1(self):
        """Test SNP and insertion."""
        result = snp_mutation("AAATTTCCCGGG", 2, "A", "T")
        # result: ATATTTCCCGGG
        result = insertion_mutation(result, 4, "T", "TCGCG")
        # result: ATAT>>>>CGCGTTCCCGGG
        result = snp_mutation(result, 4, "T", "G")
        # result: ATAG>>>>CGCGTTCCCGGG
        result = insertion_mutation(result, 4, "G", "GTTT")
        # result: ATAG>>>TTT>>>>CGCGTTCCCGGG
        result = insertion_mutation(result, 19, "T", "TAATAT")
        # result: ATAG>>>TTT>>>>CGCGT>>>>>AATATTCCCGGG
        assert result == "ATAG>>>TTT>>>>CGCGT>>>>>AATATTCCCGGG"

    def test_mixed_2(self):
        """Test SNP and deletion."""
        result = deletion_mutation("CGATCGGATACTTACTTT", 1, "CGAT", "C")
        # result: C<<<CGGATACTTACTTT
        result = snp_mutation(result, 1, "C", "T")
        # result: T<<<CGGATACTTACTTT
        result = snp_mutation(result, 5, "C", "G")
        # result: T<<<GGGATACTTACTTT
        result = deletion_mutation(result, 5, "GGGATA", "G")
        # result: T<<<G<<<<<CTTACTTT
        assert result == "T<<<G<<<<<CTTACTTT"

    def test_mixed_3(self):
        """Test insertion and deletion."""
        result = deletion_mutation("GCCTATTGGTATACTN", 5, "ATTGG", "A")
        # result: GCCTA<<<<TATACTN
        result = deletion_mutation(result, 10, "TATA", "T")
        # result: GCCTA<<<<T<<<CTN
        result = insertion_mutation(result, 1, "G", "GTA")
        # result: G>>TACCTA<<<<T<<<CTN
        result = insertion_mutation(result, 18, "C", "CCC")
        # result: G>>TACCTA<<<<T<<<C>>CCTN
        result = deletion_mutation(result, 22, "CT", "T")
        # result: G>>TACCTA<<<<T<<<C>>CC<N
        assert result == "G>>TACCTA<<<<T<<<C>>CC<N"

    def test_mixed_4(self):
        """Test SNP, insertion and deletion."""
        result = snp_mutation("TAGGTATAGTTCCGAT", 4, "G", "C")
        # result: TAGCTATAGTTCCGAT
        result = insertion_mutation(result, 6, "A", "ACCC")
        # result: TAGCTA>>>CCCTAGTTCCGAT
        result = deletion_mutation(result, 10, "CCCTAG", "C")
        # result: TAGCTA>>>C<<<<<TTCCGAT
        result = snp_mutation(result, 20, "G", "C")
        # result: TAGCTA>>>C<<<<<TTCCCAT
        result = insertion_mutation(result, 21, "A", "ATAT")
        # result: TAGCTA>>>C<<<<<TTCCCA>>>TATT
        result = deletion_mutation(result, 27, "TT", "T")
        # result: TAGCTA>>>C<<<<<TTCCCA>>>TAT<
        assert result == "TAGCTA>>>C<<<<<TTCCCA>>>TAT<"
