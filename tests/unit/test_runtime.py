import os
import logging
from unittest.mock import patch, MagicMock
import tensorflow as tf

from panthera.core.runtime import (
    initialize_runtime,
    _configure_suppressions,
    _configure_tensorflow_behavior,
    _setup_gpu_memory,
)


def test_initialize_runtime_success():
    """Test standard execution of initialize_runtime."""
    with (
        patch("panthera.core.runtime._configure_suppressions") as mock_sup,
        patch("panthera.core.runtime._configure_tensorflow_behavior") as mock_tf,
        patch("panthera.core.runtime._setup_gpu_memory") as mock_gpu,
    ):
        mock_gpu.return_value = {"device": "CPU", "count": 0}

        result = initialize_runtime(silent=True, use_mixed_precision=False)

        mock_sup.assert_called_once()
        mock_tf.assert_called_once_with(False)
        mock_gpu.assert_called_once_with(True)
        assert result == {"device": "CPU", "count": 0}


def test_configure_suppressions():
    """Test that suppressions configure environment variables and warnings."""
    # Reset env to ensure test isolation
    if "TF_CPP_MIN_LOG_LEVEL" in os.environ:
        del os.environ["TF_CPP_MIN_LOG_LEVEL"]

    _configure_suppressions()

    assert os.environ.get("TF_CPP_MIN_LOG_LEVEL") == "3"
    # Note: Checking the exact internal state of warnings can be complex,
    # but asserting it runs without error is a good baseline.
    assert tf.get_logger().level == logging.ERROR


@patch("tensorflow.keras.mixed_precision.Policy")
@patch("tensorflow.keras.mixed_precision.set_global_policy")
@patch("tensorflow.config.optimizer.set_jit")
def test_configure_tensorflow_behavior_enabled(
    mock_set_jit, mock_set_global, mock_policy
):
    """Test TF configuration when mixed precision is enabled."""
    _configure_tensorflow_behavior(use_mixed_precision=True)

    mock_policy.assert_called_once_with("mixed_float16")
    mock_set_global.assert_called_once()
    mock_set_jit.assert_called_once_with(True)


@patch("tensorflow.keras.mixed_precision.Policy")
def test_configure_tensorflow_behavior_disabled(mock_policy):
    """Test TF configuration does nothing when disabled."""
    _configure_tensorflow_behavior(use_mixed_precision=False)
    mock_policy.assert_not_called()


@patch("tensorflow.keras.mixed_precision.Policy")
def test_configure_tensorflow_behavior_exception(mock_policy, caplog):
    """Test TF configuration gracefully handles exceptions."""
    mock_policy.side_effect = Exception("Mocked hardware error")

    with caplog.at_level(logging.WARNING):
        _configure_tensorflow_behavior(use_mixed_precision=True)

    assert (
        "Could not enable hardware acceleration: Mocked hardware error" in caplog.text
    )


@patch("tensorflow.config.list_physical_devices")
def test_setup_gpu_memory_no_gpus_silent(mock_list_devices, caplog):
    """Test GPU memory setup when no GPUs are found and silent=True."""
    mock_list_devices.return_value = []

    with caplog.at_level(logging.WARNING):
        result = _setup_gpu_memory(silent=True)

    assert result == {"device": "CPU", "count": 0}
    assert "No GPU detected" not in caplog.text


@patch("tensorflow.config.list_physical_devices")
def test_setup_gpu_memory_no_gpus_verbose(mock_list_devices, caplog):
    """Test GPU memory setup when no GPUs are found and silent=False."""
    mock_list_devices.return_value = []

    with caplog.at_level(logging.WARNING):
        result = _setup_gpu_memory(silent=False)

    assert result == {"device": "CPU", "count": 0}
    assert "No GPU detected" in caplog.text


@patch("tensorflow.config.list_physical_devices")
@patch("tensorflow.config.experimental.set_memory_growth")
def test_setup_gpu_memory_with_gpus(mock_set_memory, mock_list_devices):
    """Test GPU memory setup allocates memory across available GPUs."""
    mock_gpu_1 = MagicMock()
    mock_gpu_2 = MagicMock()
    mock_list_devices.return_value = [mock_gpu_1, mock_gpu_2]

    result = _setup_gpu_memory(silent=True)

    assert result == {"device": "GPU", "count": 2, "details": [mock_gpu_1, mock_gpu_2]}
    assert mock_set_memory.call_count == 2
    mock_set_memory.assert_any_call(mock_gpu_1, True)
    mock_set_memory.assert_any_call(mock_gpu_2, True)


@patch("tensorflow.config.list_physical_devices")
@patch("tensorflow.config.experimental.set_memory_growth")
def test_setup_gpu_memory_runtime_error(mock_set_memory, mock_list_devices, caplog):
    """Test GPU memory setup handles RuntimeError correctly."""
    mock_gpu = MagicMock()
    mock_list_devices.return_value = [mock_gpu]
    mock_set_memory.side_effect = RuntimeError("Device already initialized")

    with caplog.at_level(logging.ERROR):
        result = _setup_gpu_memory(silent=True)

    assert result == {"device": "GPU_ERROR", "error": "Device already initialized"}
    assert "Critical: GPU initialization failed" in caplog.text
