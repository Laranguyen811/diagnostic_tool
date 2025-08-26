import numpy as np
import pytest
import math

from diagnostic_tool.biodiversity_metrics import calculate_shannon_wiener_index_batch,calculate_biodiversity_units, calculate_species_richness,calculate_habitat_condition_score

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
    with pytest.raises(ValueError):
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

def test_condition_validate_input():
    condition_data = {
        "vegetation_cover":90.0,
        "soil_quality": 0.8,
        "water_quality": 0.9,
        "invasive_species": 0.1,
        "fauna_diversity": 0.7,
    }
    result = calculate_habitat_condition_score(condition_data)
    expected = (90/100) * 0.8 * 0.9 * (1 - 0.1) * 0.7
    assert result == expected

def test_condition_out_of_range():
    condition_data = {
        "vegetation_cover": 110,  # Invalid
        "soil_quality": 0.8,
        "water_quality": 0.9,
        "invasive_species": 0.1,
        "fauna_diversity": 0.7,
    }
    with pytest.raises(ValueError) as excinfo:
        calculate_habitat_condition_score(condition_data)
    assert "vegetation_cover must be between 0.0 and 100.0" in str(excinfo.value)

def test_condition_missing_key():
    condition_data = {
        "vegetation_cover": 90,
        "soil_quality": 0.8,
        # "water_quality" missing
        "invasive_species": 0.1,
        "fauna_diversity": 0.7,
    }
    with pytest.raises(ValueError) as excinfo:
        calculate_habitat_condition_score(condition_data)

def test_condition_type_mismatch():
    condition_data = {
        "vegetation_cover": "high",  # Invalid type
        "soil_quality": 0.8,
        "water_quality": 0.9,
        "invasive_species": 0.1,
        "fauna_diversity": 0.7,
    }
    with pytest.raises(TypeError) as excinfo:
        calculate_habitat_condition_score(condition_data)

    print(f"Raised error: {excinfo.value}")

    assert "vegetation_cover must be of type float or int" in str(excinfo.value)

@pytest.mark.parametrize("veg_cover",[0.0,100.0])
def test_vegetation_cover_boundaries(veg_cover):
    condition_data = {
        "vegetation_cover": veg_cover,
        "soil_quality": 1.0,
        "water_quality": 1.0,
        "invasive_species": 0.0,
        "fauna_diversity": 1.0,
    }
    result = calculate_habitat_condition_score(condition_data)
    expected = (veg_cover/100)
    assert result == expected

def test_condition_nan_input():
    condition_data = {
        "vegetation_cover": 90,
        "soil_quality": float('nan'),  # Invalid
        "water_quality": 0.9,
        "invasive_species": 0.1,
        "fauna_diversity": 0.7,
    }
    with pytest.raises(ValueError) as excinfo:
        calculate_habitat_condition_score(condition_data)
    assert "Soil quality must be between 0.0 and 1.0" in str(excinfo.value)

def test_condition_extra_keys():
    condition_data = {
        "vegetation_cover": 90,
        "soil_quality": 0.8,
        "water_quality": 0.9,
        "invasive_species": 0.1,
        "fauna_diversity": 0.7,
        "extra_param": 42,  # Extra key
    }
    result = calculate_habitat_condition_score(condition_data)
    assert isinstance(result,float)

def test_condition_all_zeros():
    condition_data = {
        "vegetation_cover": 0.0,
        "soil_quality": 0.0,
        "water_quality": 0.0,
        "invasive_species": 0.0,
        "fauna_diversity": 0.0,
    }
    result = calculate_habitat_condition_score(condition_data)
    assert result == 0.0



