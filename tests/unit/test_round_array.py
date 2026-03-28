import pytest
import numpy as np
from panthera.core.ssp.predict import round_array


def test_round_array_basic():
    """Test standard rounding to default (5) decimals."""
    input_data = [
        np.array([1.123456789, 2.678912345], dtype=np.float32),
        np.array([0.000123456, 0.999987654], dtype=np.float32),
    ]
    expected = [
        np.array([1.12346, 2.67891], dtype=np.float32),
        np.array([0.00012, 0.99999], dtype=np.float32),
    ]

    result = round_array(input_data)

    for res, exp in zip(result, expected):
        np.testing.assert_array_equal(res, exp)


def test_round_array_custom_decimals():
    """Test rounding to a specific number of decimal places."""
    input_data = [np.array([1.555, 2.444], dtype=np.float32)]
    decimals = 1
    expected = [np.array([1.6, 2.4], dtype=np.float32)]

    result = round_array(input_data, decimals=decimals)

    np.testing.assert_array_equal(result[0], expected[0])


def test_round_array_empty_list():
    """Test that an empty list returns an empty list."""
    assert round_array([]) == []


def test_round_array_negative_numbers():
    """Test rounding with negative values."""
    input_data = [np.array([-1.1234, -2.5678], dtype=np.float32)]
    expected = [np.array([-1.123, -2.568], dtype=np.float32)]

    result = round_array(input_data, decimals=3)

    np.testing.assert_array_equal(result[0], expected[0])


def test_round_array_immutability():
    """
    Ensure the original arrays are not modified in place
    (standard np.round behavior).
    """
    original_array = np.array([1.12345], dtype=np.float32)
    input_data = [original_array]

    round_array(input_data, decimals=1)

    # Check that original_array still has full precision
    assert original_array[0] == np.float32(1.12345)


@pytest.mark.parametrize(
    "decimals, expected_val",
    [
        (0, 1.0),
        (2, 1.12),
    ],
)
def test_round_array_parameterized(decimals, expected_val):
    """Quick check across different rounding scales."""
    data = [np.array([1.1234], dtype=np.float32)]
    result = round_array(data, decimals=decimals)
    assert result[0][0] == np.float32(expected_val)
