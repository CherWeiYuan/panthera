import pytest
import tensorflow as tf
from pathlib import Path
from google.protobuf.message import DecodeError

# Import the functions we wrote earlier
from panthera.core.ml.load_model import wrap_frozen_graph, load_frozen_graph

# ==========================================
# Fixtures for Test Setup
# ==========================================

@pytest.fixture
def dummy_graph_def():
    """
    Creates a minimal valid TF1 GraphDef in memory.
    Architecture: Takes an input 'x', multiplies it by 2.0, outputs 'y'.
    """
    with tf.Graph().as_default() as g:
        # Define input and output tensors explicitly
        x = tf.compat.v1.placeholder(tf.float32, shape=[], name="input_x")
        multiplier = tf.constant(2.0, dtype=tf.float32)
        tf.multiply(x, multiplier, name="output_y")
        
    return g.as_graph_def()

@pytest.fixture
def valid_pb_file(tmp_path, dummy_graph_def):
    """
    Writes the dummy GraphDef to a temporary .pb file that cleans itself up.
    """
    pb_path = tmp_path / "dummy_model.pb"
    with open(pb_path, "wb") as f:
        f.write(dummy_graph_def.SerializeToString())
    return pb_path

@pytest.fixture
def corrupted_pb_file(tmp_path):
    """
    Creates a fake/corrupted .pb file containing raw text.
    """
    pb_path = tmp_path / "corrupted_model.pb"
    with open(pb_path, "w") as f:
        f.write("This is not a valid protobuf binary file.")
    return pb_path


# ==========================================
# Test Cases: wrap_frozen_graph
# ==========================================

def test_wrap_frozen_graph_success(dummy_graph_def):
    """Test that a valid graph is wrapped and computes correctly."""
    frozen_func = wrap_frozen_graph(
        graph_def=dummy_graph_def,
        inputs=["input_x:0"],
        outputs=["output_y:0"]
    )
    
    # Verify it's callable and produces the expected mathematical result (x * 2.0)
    input_tensor = tf.constant(5.0)
    result = frozen_func(input_tensor)
    
    assert result[0].numpy() == 10.0

def test_wrap_frozen_graph_invalid_tensors(dummy_graph_def):
    """Test that passing wrong tensor names raises a ValueError."""
    with pytest.raises(ValueError, match="Tensor not found in graph"):
        wrap_frozen_graph(
            graph_def=dummy_graph_def,
            inputs=["wrong_input_name:0"],
            outputs=["output_y:0"]
        )


# ==========================================
# Test Cases: load_frozen_graph
# ==========================================

def test_load_frozen_graph_success(valid_pb_file):
    """Test loading a legitimate .pb file from disk."""
    frozen_func = load_frozen_graph(
        graph_filepath=valid_pb_file,
        inputs=["input_x:0"],
        outputs=["output_y:0"]
    )
    
    # Test inference to ensure the graph loaded properly
    input_tensor = tf.constant(3.0)
    result = frozen_func(input_tensor)
    
    assert result[0].numpy() == 6.0

def test_load_frozen_graph_file_not_found():
    """Test behavior when the .pb file does not exist."""
    fake_path = Path("does_not_exist/model.pb")
    
    with pytest.raises(FileNotFoundError, match="Frozen graph file not found"):
        load_frozen_graph(
            graph_filepath=fake_path,
            inputs=["input_x:0"],
            outputs=["output_y:0"]
        )

def test_load_frozen_graph_decode_error(corrupted_pb_file):
    """Test behavior when the .pb file is corrupted or invalid."""
    with pytest.raises(DecodeError):
        load_frozen_graph(
            graph_filepath=corrupted_pb_file,
            inputs=["input_x:0"],
            outputs=["output_y:0"]
        )
