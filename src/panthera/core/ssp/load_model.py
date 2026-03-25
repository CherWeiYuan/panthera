import logging
from pathlib import Path
from typing import List, Callable, Union

import tensorflow as tf
from google.protobuf.message import DecodeError

# Configure module-level logger
logger = logging.getLogger(__name__)


def wrap_frozen_graph(
    graph_def: tf.compat.v1.GraphDef, inputs: List[str], outputs: List[str]
) -> Callable:
    """
    Wraps a TensorFlow 1.x GraphDef into a TensorFlow 2.x ConcreteFunction.

    Args:
        graph_def: The parsed TensorFlow frozen graph definition.
        inputs: A list of string names for the input tensors (e.g., ["input:0"]).
        outputs: A list of string names for the output tensors (e.g., ["output:0"]).

    Returns:
        A TensorFlow ConcreteFunction that can be called with input tensors.

    Raises:
        ValueError: If specified inputs or outputs are not found in the graph.
    """

    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    try:
        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph

        # Prune the graph to extract only the specified inputs and outputs
        frozen_func = wrapped_import.prune(
            tf.nest.map_structure(import_graph.as_graph_element, inputs),
            tf.nest.map_structure(import_graph.as_graph_element, outputs),
        )
        return frozen_func

    except KeyError as e:
        logger.error(f"Failed to find specified tensor in graph: {e}")
        raise ValueError(
            f"Tensor not found in graph. Verify your inputs {inputs} and outputs {outputs}."
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error while wrapping frozen graph: {e}")
        raise


def load_frozen_graph(
    graph_filepath: Union[str, Path],
    inputs: tuple[str] = ("x:0",),
    outputs: tuple[str] = ("Identity:0",),
) -> Callable:
    """
    Loads a frozen TensorFlow graph from disk and wraps it into a callable function.

    Args:
        graph_filepath: Path to the .pb frozen graph file.
        inputs: List of input tensor names. Default: ["x:0"]
        outputs: List of output tensor names. Default: ["Identity:0"]

    Returns:
        A callable TensorFlow ConcreteFunction for inference.

    Raises:
        FileNotFoundError: If the graph_filepath does not exist.
        DecodeError: If the file is not a valid protobuf or is corrupted.
    """
    graph_path = Path(graph_filepath)

    if not graph_path.is_file():
        logger.error(f"Graph file not found at: {graph_path}")
        raise FileNotFoundError(f"Frozen graph file not found at {graph_path}")

    logger.info(f"Loading frozen graph from {graph_path}...")

    try:
        with tf.io.gfile.GFile(str(graph_path), "rb") as handler:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(handler.read())

    except DecodeError as e:
        logger.error(
            f"Failed to parse the graph file. It may be corrupted or not a valid .pb file: {e}"
        )
        raise
    except Exception as e:
        logger.error(f"Error reading the graph file {graph_path}: {e}")
        raise

    logger.debug(
        f"Successfully loaded GraphDef. Wrapping into ConcreteFunction with inputs={inputs}, outputs={outputs}"
    )

    return wrap_frozen_graph(
        graph_def=graph_def,
        inputs=list(inputs),
        outputs=list(outputs),
    )
