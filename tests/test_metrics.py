import warnings

import numpy as np
import pytest
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from statsmodels.graphics.tukeyplot import results
import logging
import sys

from diagnostic_tool.biodiversity_metrics import calculate_shannon_wiener_index_batch,calculate_biodiversity_units, calculate_species_richness,calculate_habitat_condition_score, calculate_endemism_index, calculate_functional_richness, calculate_simpson_index


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    stream=sys.stdout  # Ensures logs go to console, not stderr
)

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

def test_endemism_index_validate_input():
    endemism_data = [
        {"presence_or_absence": 1, "total_regions": 10},
        {"presence_or_absence": 0, "total_regions": 10},
        {"presence_or_absence": 1, "total_regions": 10},
    ]
    result = calculate_endemism_index(endemism_data)
    expected = 0.2
    assert result == expected

def test_endemism_index_out_of_range():
    endemism_data = [
        {"presence_or_absence":2, "total_regions": 10}, # out of range
        {"presence_or_absence":0, "total_regions": 10},
        {"presence_or_absence":1, "total_regions": 10},
    ]
    with pytest.raises(ValueError) as excinfo:
        calculate_endemism_index(endemism_data)
    print (excinfo.value)
    assert "presence_or_absence must be between 0 and 1" in str(excinfo.value)

def test_endemism_missing_key():
    endemism_data = [
        {"presence_or_absence": 1, "total_regions": 10},
        {"total_regions": 10},  # Missing presence_or_absence
    ]
    with pytest.raises(ValueError) as excinfo:
        calculate_endemism_index(endemism_data)
    print (excinfo.value)

def test_endemism_type_mismatch():
    endemism_data = [
        {"presence_or_absence": 1, "total_regions": "10"}, # Type mismatch
        {"presence_or_absence": 0, "total_regions": 100},
        {"presence_or_absence": 1, "total_regions": 10},
        {"presence_or_absence":None, "total_regions": 10}, # Type mismatch
    ]
    with pytest.raises(TypeError) as excinfo:
        calculate_endemism_index(endemism_data)
    print (excinfo.value)

def test_endemism_zero_total():
    endemism_data = [
        {"presence_or_absence":0, "total_regions": 0}, # Total regions zero
        {"presence_or_absence":0, "total_regions": 10}, # Total regions zero
    ]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = calculate_endemism_index(endemism_data)
        assert result == (0.0, 1)
        assert any("Skipping invalid record" in str(warn.message) for warn in w)


def test_endemism_empty_input():
    endemism_data = []
    result = calculate_endemism_index(endemism_data)
    assert result == (0.0, 0)

def test_endemism_mixed_validity():
    endemism_data = [
        {"presence_or_absence": 1, "total_regions": 10},
        {"presence_or_absence": 0, "total_regions": 10},
        {"presence_or_absence": 1, "total_regions": 10},
        {"presence_or_absence": 2, "total_regions": 10}, # Invalid inputs

    ]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = calculate_endemism_index(endemism_data)
        assert result == (0.2, 1)
        assert any("Skipping invalid record" in str(warn.message) for warn in w)


def test_endemism_precision():
    endemism_data = [
        {"presence_or_absence": 1, "total_regions": 3},
        {"presence_or_absence": 1, "total_regions": 3},
    ]
    result = calculate_endemism_index(endemism_data)
    assert pytest.approx(result[0], 0.01) == 0.6667

def test_functional_richness_basic():
    trait_data : List[Dict[str, Any]] = [
        {"species_id": "A", "abundance": 10.0, "trait_1": "fast", "trait_2": 0.5, "trait_3": "small", "trait_4": 1.0,
         "trait_5": "red", "trait_6": 2.0},
        {"species_id": "B", "abundance": 20.0, "trait_1": "slow", "trait_2": 0.7, "trait_3": "large", "trait_4": 1.5,
         "trait_5": "blue", "trait_6": 2.5},
        {"species_id": "C", "abundance": 15.0, "trait_1": "medium", "trait_2": 0.6, "trait_3": "medium", "trait_4": 1.2,
         "trait_5": "green", "trait_6": 2.2},
    ]
    richness = calculate_functional_richness(trait_data,trait_count=6)
    assert richness == 0.0

def test_functional_richness_missing_key():
    trait_data : List[Dict[str, Any]] = [
        {"species_id": "A", "abundance": 10.0, "trait_1": "fast", "trait_2": 0.5, "trait_3": "small", "trait_4": 1.0,
         "trait_5": "red"},
        {"species_id": "B", "abundance": 20.0, "trait_1": "slow", "trait_2": 0.7, "trait_3": "large", "trait_4": 1.5,
         "trait_5": "blue", "trait_6": 2.5},
    ]
    with pytest.raises(ValueError) as excinfo:
        calculate_functional_richness(trait_data,trait_count=6)
    print (excinfo.value)

def test_functional_richness_type_mismatch():
    trait_data : List[Dict[str, Any]] = [
        {"species_id": "A", "abundance": "a lot", "trait_1": "fast", "trait_2": 0.5, "trait_3": "small", "trait_4": 1.0,
         "trait_5": "red", "trait_6": 2.0},
        {"species_id": "B", "abundance": 20.0, "trait_1": "slow", "trait_2": 0.7, "trait_3": "large", "trait_4": 1.5,
         "trait_5": "blue", "trait_6": 2.5},
    ]
    with pytest.raises(TypeError) as excinfo:
        calculate_functional_richness(trait_data,trait_count=6)
    print (excinfo.value)

def test_functional_richness_empty_input():
    trait_data : List[Dict[str, Any]] = []
    with pytest.raises(KeyError) as excinfo:
        calculate_functional_richness(trait_data,trait_count=6)
    print (excinfo.value)

def test_functional_richness_mixed_validity():
    trait_data : List[Dict[str, Any]] = [
        {"species_id": "A", "abundance": 10.0, "trait_1": "fast", "trait_2": 0.5, "trait_3": "small", "trait_4": 1.0,
         "trait_5": "red", "trait_6": 2.0},
        {"species_id": "B", "abundance": -20.0, "trait_1": "slow", "trait_2": 0.7, "trait_3": "large", "trait_4": 1.5,
         "trait_5": "blue", "trait_6": 2.5}, # Invalid abundance
        {"species_id": "C", "abundance": 15.0, "trait_1": "medium", "trait_2": 0.6, "trait_3": "medium", "trait_4": 1.2,
         "trait_5": "green", "trait_6": 2.2},
    ]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        richness = calculate_functional_richness(trait_data,trait_count=6)
        assert richness == 0.0
        assert any("negative abundance" in str(warn.message) for warn in w)

def test_functional_richness_high_precision():
    trait_data : List[Dict[str, Any]] = [
        {"species_id": "A", "abundance": 1e-10, "trait_1": "fast", "trait_2": 0.5, "trait_3": "small", "trait_4": 1.0,
         "trait_5": "red", "trait_6": 2.0},
        {"species_id": "B", "abundance": 2e-10, "trait_1": "slow", "trait_2": 0.7, "trait_3": "large", "trait_4": 1.5,
         "trait_5": "blue", "trait_6": 2.5},
    ]
    richness = calculate_functional_richness(trait_data,trait_count=6)
    assert round(richness, 10) == 0.0

def test_functional_richness_nonzero_volume(caplog):
    caplog.set_level(logging.INFO, logger="diagnostic_tool.biodiversity_metrics")
    trait_data = [
        {"species_id": "A", "abundance": 10.0, "trait_1": "fast", "trait_2": 0.1, "trait_3": "small", "trait_4": 0.1,
         "trait_5": "red", "trait_6": 0.1},
        {"species_id": "B", "abundance": 20.0, "trait_1": "slow", "trait_2": 0.9, "trait_3": "large", "trait_4": 0.9,
         "trait_5": "blue", "trait_6": 0.9},
        {"species_id": "C", "abundance": 15.0, "trait_1": "medium", "trait_2": 0.5, "trait_3": "medium", "trait_4": 0.5,
         "trait_5": "green", "trait_6": 0.5},
    ]
    richness = calculate_functional_richness(trait_data,trait_count=6)
    assert richness == 0.0
    assert any("insufficient rank" in message for message in caplog.messages)
    print(caplog.text)

def test_functional_richness_all_traits_missing():
    trait_data = [
        {"species_id": "A", "abundance": 10.0},
        {"species_id": "B", "abundance": 20.0},
    ]
    with pytest.raises(ValueError) as excinfo:
        calculate_functional_richness(trait_data,trait_count=6)
    print(excinfo.value)

def test_simpson_index_basic():
    simpson_data = [
        {"species_id":"A", "abundance": 10},
        {"species_id":"B", "abundance": 20},
        {"species_id":"C", "abundance": 30},

    ]

    result = calculate_simpson_index(simpson_data)
    total = 10 + 20 + 30
    expected = 1- ((10/total)**2 + (20/total)**2 + (30/total)**2)
    assert np.isclose(result,expected)
