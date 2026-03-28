import os
import logging
import warnings

# Set up module-level logging
logger = logging.getLogger(__name__)


def initialize_runtime(silent: bool = False, use_mixed_precision: bool = True):
    """
    Standardizes the environment and hardware state for Panthera.
    Returns: A dictionary of detected hardware capabilities.
    """
    # Suppress OS warnings
    _configure_suppressions()

    # Import tensorflow after OS logs and warnings are suppressed
    # to avoid printing TF warnings
    import tensorflow as tf

    _configure_tensorflow_behavior(tf, use_mixed_precision)
    gpu_metadata = _setup_gpu_memory(tf, silent)

    logger.debug("Runtime environment successfully initialized.")
    return gpu_metadata


def _configure_suppressions():
    """Handles warnings and external library noise."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings("ignore", category=UserWarning)


def _configure_tensorflow_behavior(tf, use_mixed_precision: bool):
    """Sets performance optimizations like XLA and Mixed Precision."""
    # Silencing TF internal loggers
    tf.get_logger().setLevel(logging.ERROR)

    if use_mixed_precision:
        try:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
            # XLA (Accelerated Linear Algebra) compilation
            tf.config.optimizer.set_jit(True)
            logger.debug("Performance: Mixed Precision (FP16) and XLA enabled.")
        except Exception as e:
            logger.warning(f"Could not enable hardware acceleration: {e}")


def _setup_gpu_memory(tf, silent: bool):
    """Manages VRAM allocation and device detection."""
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        if not silent:
            logger.warning("No GPU detected. Falling back to CPU execution.")
        return {"device": "CPU", "count": 0}

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.debug(f"Memory growth enabled for {len(gpus)} GPU(s).")
        return {"device": "GPU", "count": len(gpus), "details": gpus}
    except RuntimeError as e:
        # This usually happens if TF was initialized before this function was called
        logger.error(
            f"Critical: GPU initialization failed. Device already in use? | {e}"
        )
        return {"device": "GPU_ERROR", "error": str(e)}
