import pytest
import numpy as np
import numpy.testing as npt

# Assuming your class is in a file named encoder.py
from panthera.core.ssp.onehotencoder import SeqEncoder, EncodingSchema


class TestGenomicEncoder:
    """Test suite for the GenomicEncoder class."""

    def test_spliceai_basic_encoding(self):
        """Test standard DNA/RNA characters using the SPLICEAI schema."""
        sequence = "ACGT"
        expected = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # A
                [0.0, 1.0, 0.0, 0.0],  # C
                [0.0, 0.0, 1.0, 0.0],  # G
                [0.0, 0.0, 0.0, 1.0],  # T
            ],
            dtype=np.float32,
        )

        result = SeqEncoder.one_hot_encode(sequence, EncodingSchema.SPLICEAI)
        npt.assert_array_equal(result, expected)

    def test_modelp_basic_encoding(self):
        """Test standard DNA/RNA characters using the MODELP schema to ensure
        mapping differences.
        """
        sequence = "ACGT"
        expected = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # A
                [0.0, 0.0, 1.0, 0.0],  # C (Note difference from SpliceAI)
                [0.0, 0.0, 0.0, 1.0],  # G (Different)
                [0.0, 1.0, 0.0, 0.0],  # T (Different)
            ],
            dtype=np.float32,
        )

        result = SeqEncoder.one_hot_encode(sequence, EncodingSchema.MODELP)
        npt.assert_array_equal(result, expected)

    @pytest.mark.parametrize("schema", [EncodingSchema.SPLICEAI, EncodingSchema.MODELP])
    def test_case_insensitivity(self, schema):
        """Ensure lowercase and uppercase sequences produce identical output."""
        upper_result = SeqEncoder.one_hot_encode("ACGTUNX", schema)
        lower_result = SeqEncoder.one_hot_encode("acgtunx", schema)

        npt.assert_array_equal(upper_result, lower_result)

    @pytest.mark.parametrize("schema", [EncodingSchema.SPLICEAI, EncodingSchema.MODELP])
    def test_output_properties(self, schema):
        """Validate the returned object is a NumPy
        array of the correct shape and type.
        """
        sequence = "AGCT"
        result = SeqEncoder.one_hot_encode(sequence, schema)

        assert isinstance(result, np.ndarray), "Output must be a NumPy array."
        assert result.dtype == np.float32, (
            "Output datatype must be float32 for memory efficiency."
        )
        assert result.shape == (len(sequence), 4), (
            f"Expected shape {(len(sequence), 4)}, got {result.shape}"
        )

    @pytest.mark.parametrize("schema", [EncodingSchema.SPLICEAI, EncodingSchema.MODELP])
    def test_empty_sequence(self, schema):
        """Test boundary condition: empty string
        should return an empty array with 4 columns.
        """
        result = SeqEncoder.one_hot_encode("", schema)

        assert result.shape == (0, 4)
        assert len(result) == 0

    def test_invalid_character_raises_value_error(self):
        """Ensure passing invalid characters (like Z)
        raises a descriptive ValueError.
        """
        invalid_sequence = "ACGTZ"

        # We expect a ValueError to be raised
        with pytest.raises(ValueError) as exc_info:
            SeqEncoder.one_hot_encode(invalid_sequence, EncodingSchema.SPLICEAI)

        # Verify the error message is helpful
        assert "Invalid character 'Z'" in str(exc_info.value)

    @pytest.mark.parametrize("schema", [EncodingSchema.SPLICEAI, EncodingSchema.MODELP])
    def test_uracil_encoding(self, schema):
        """Uracil (U) should be encoded exactly like Thymine (T)."""
        seq_t = SeqEncoder.one_hot_encode("T", schema)
        seq_u = SeqEncoder.one_hot_encode("U", schema)
        npt.assert_array_equal(seq_t, seq_u)

    @pytest.mark.parametrize("schema", [EncodingSchema.SPLICEAI, EncodingSchema.MODELP])
    def test_single_character(self, schema):
        """Test encoding a sequence of exactly length 1."""
        result = SeqEncoder.one_hot_encode("A", schema)
        assert result.shape == (1, 4)

    @pytest.mark.parametrize("schema", [EncodingSchema.SPLICEAI, EncodingSchema.MODELP])
    def test_whitespace_handling(self, schema):
        """Whitespace should be stripped seamlessly before encoding."""
        clean_result = SeqEncoder.one_hot_encode("ACGT", schema)
        messy_result = SeqEncoder.one_hot_encode("  ACGT\n\t ", schema)
        npt.assert_array_equal(clean_result, messy_result)
