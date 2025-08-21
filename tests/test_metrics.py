import numpy as np
import pytest
import math

from diagnostic_tool.biodiversity_metrics import calculate_shannon_wiener_index_batch,calculate_biodiversity_units, calculate_species_richness

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


def test_bio_units_validate_input():
    unit_data ={
        "area": 10.0,
        "distinctiveness": 0.8,
        "condition": 0.9,
        "strategic_significance": 1.0,
        "connectivity": 0.7,
    }
    results = calculate_biodiversity_units(unit_data)
    expected = 10.0 * 0.8 * 0.9 * 1.0 * 0.7
    assert results == expected

def test_required_keys():
    unit_data = {
        "area": 10.0,
        "distinctiveness": 0.8,
        "condition": 0.9,
        # strategic_significance missing
        "connectivity": 0.7,

    }
    with pytest.raises(ValueError) as excinfo:
        calculate_biodiversity_units(unit_data)
    assert "Missing required keys" in str(excinfo.value)

def test_bio_unit_out_of_range_score():
    unit_data = {
        "area": 10.0,
        "distinctiveness": 1.2,  # Invalid
        "condition": 0.9,
        "strategic_significance": 1.0,
        "connectivity": 0.7,
    }
    with pytest.raises(ValueError) as excinfo:
        calculate_biodiversity_units(unit_data)
    assert "Distinctiveness must be between 0.0 and 1.0" in str(excinfo.value)

def test_bio_unit_edge_case():
    unit_data = {
        "area": 0.0,
        "distinctiveness": 0.8,
        "condition": 0.9,
        "strategic_significance": 1.0,
        "connectivity": 0.7,
    }

    with pytest.raises(ValueError) as excinfo:
        calculate_biodiversity_units(unit_data)
    assert "Area must be greater than 0" in str(excinfo.value)

def test_basic_richness():
    data = {"total_species": 100, "area": 50.0}
    assert calculate_species_richness(data) == 2.0

def test_high_precision():
    data = {"total_species": 123, "area": 0.0001}
    richness = calculate_species_richness(data)
    assert round(richness, 2) == 1230000.00

def test_zero_area_strict():
    data = {"total_species": 100, "area": 0.0}
    with pytest.raises(ValueError):
        calculate_species_richness(data,strict=True)

def test_zero_area_non_strict():
    data = {"total_species": 100, "area": 0.0}
    assert math.isnan(calculate_species_richness(data,strict=False))

def test_negative_species():
    data = {"total_species": -5, "area": 10.0}
    assert math.isnan(calculate_species_richness(data,strict=False))

def test_missing_keys():
    data = {"area": 10.0}
    with pytest.raises(KeyError):
        calculate_species_richness(data)

def test_type_mismatch():
    data = {"total_species": "a lot", "area": "big"}
    with pytest.raises(TypeError) as excinfo:
        calculate_species_richness(data)
    print(excinfo.value)

def test_extremely_large_area():
    data = {"total_species": 100, "area": 1e12}
    richness = calculate_species_richness(data)
    assert math.isclose(richness, 1e-10,rel_tol=1e-12)

def test_species_zero():
    data = {"total_species": 0, "area": 10.0}
    assert calculate_species_richness(data) == 0.0

@pytest.mark.parametrize("species, area, expected",[
    (10, 2.0, 5.0),
    (0, 1.0, 0.0),
    (50, 0.5, 100.0),

])
def test_species_richness_cases(species, area, expected):
    data = {"total_species": species, "area": area}
    result = calculate_species_richness(data)
    assert result == expected



