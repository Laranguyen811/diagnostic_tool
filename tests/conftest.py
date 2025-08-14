import numpy as np
import pytest
from diagnostic_tool.biodiversity_metrics import calculate_shannon_wiener_index_batch

def test_shannon_index_typical_case():
    counts = [10,5,0,3]
    result = calculate_shannon_wiener_index_batch(counts,strict=False)
    expected = -sum([
        (10/18) * np.log(10/18),
        (5/18) * np.log(5/18),
        (3/18) * np.log(3/18)
    ])
    assert np.isclose(result,expected,atol=1e-6)

def test_shannon_index_strict_mode_raises():
    with pytest.raises(ValueError):
        calculate_shannon_wiener_index_batch([10,0,5],strict=True)

def test_shannon_index_zero_total():
    result = calculate_shannon_wiener_index_batch([0,0,0],strict=False)
    assert  np.isnan(result)