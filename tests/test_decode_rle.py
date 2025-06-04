import os
import sys
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from decode_RLE import rle_decode


def test_rle_decode_orientation():
    result = rle_decode("3 2 10 1", shape=(4, 4))
    expected = np.array([
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
    ], dtype=np.uint8)
    assert np.array_equal(result, expected)
