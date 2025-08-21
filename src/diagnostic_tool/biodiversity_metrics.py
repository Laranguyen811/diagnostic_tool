from typing import Union, Iterable
import numpy as np
import pandas as pd
import warnings
from utils.validation import validate_range,validate_array_values, validate_input_dict
def calculate_biodiversity_units(unit_data: dict) -> float:
    '''
    Calculates the biodiversity units based on area, distinctiveness, condition, strategic significance, and connectivity.
    Args:
        unit_data (dict) : A dictionary of biodiversity units, including area, distinctiveness,condition,strategic_significance and connectivity
    Returns:
        float: Calculated biodiversity units.
    '''
    required_keys = {
    "area": (0.01, float("inf")),
    "distinctiveness": (0.0, 1.0),
    "condition": (0.0, 1.0),
    "strategic_significance": (0.0, 1.0),
    "connectivity": (0.0, 1.0),
}

    validate_input_dict(unit_data,required_keys)
    validate_range(unit_data["area"],0.01,float("inf"),"area")
    for key in ["distinctiveness", "condition","strategic_significance", "connectivity"]:
        validate_range(unit_data[key],0.0,1.0,key)
    return (
            unit_data["area"]
            * unit_data["distinctiveness"]
            * unit_data["condition"]
            * unit_data["strategic_significance"]
            * unit_data["connectivity"]
    )

def calculate_species_richness(
        specimen_richness_data: dict,
        strict: bool = True,
) -> Union[int, float]:
    '''
    Calculates the species richness based on the total number of species and the area.
    Args:
        total_species (int): Total number of species in the area.
        area (float): Area of the habitat in hectares.
    Returns:
        Union[int, float]: Calculated species richness (species per hectare).
    '''
    required_keys = {
        "total_species": (0,float("inf")),
        "area": (0.0,float("inf")),
    }
    validate_input_dict(specimen_richness_data,required_keys)
    total_species = specimen_richness_data["total_species"]
    area = specimen_richness_data["area"]
    for key in ["total_species","area"]:
        validate_range(specimen_richness_data[key],0,float("inf"),key)
    if area == 0:
        if strict:
            raise ValueError("Area must be greater than zero for strict mode.")
        else:
        # In non-strict mode, return NaN or zero to indicate invalid calculation
            return float('nan')

    return total_species / area

def calculate_shannon_wiener_index_batch(
        species_counts: Iterable[int],
        strict: bool = True
)-> float:
    '''
    Calculates the Shannon-Wiener index (more sensitive to rare species, capturing subtle shifts, mirroring entropy-based reasoning) based on the number of individuals of each species (n_i) and the total number of individuals (N).
    Args:
        species_counts (Iterable[int]): List or array of individual counts per species
        strict (bool, optional): Whether to raise an error on invalid input. Defaults to True.

    Returns:
        float: Calculated Shannon-Wiener index.
    Example:
        >>> calculate_shannon_wiener_index_batch([10, 5, 0, 3], strict=False)
        0.979
    '''
    species_counts = np.array(species_counts)
    if not np.issubdtype(species_counts.dtype, np.integer):
        raise ValueError("All species counts must be integers.")
    if np.any(species_counts < 0):
        raise ValueError("All species counts must be non-negative.")
    N = species_counts.sum()
    if N <= 0:
        if strict:
            raise ValueError("Total number of individuals (N) must be greater than zero.")
        else:
            # In non-strict mode, return NaN or zero to indicate invalid calculation
            warnings.warn("N is zero - returning NaN.")
            return float('nan')

    proportions = species_counts / N
    valid_mask = proportions > 0
    if strict and not np.all(valid_mask):
        raise ValueError("Zero count species encountered in strict mode.")

    # Compute index only for valid proportions
    valid_props = proportions[valid_mask]
    shannon_wiener_index = -np.sum(valid_props * np.log(valid_props))

    return shannon_wiener_index

def calculate_habitat_condition_score(
    vegetation_cover: float,
    soil_quality: float,
    water_quality: float,
    invasive_species: float,
    fauna_diversity: float,
) -> float:
    '''
    Calculates the habitat condition score based on vegetation cover, soil quality, water quality, invasive species presence, and fauna diversity.
    Args:
        vegetation_cover (float): Percentage of vegetation cover (0-100).
        soil_quality (float): Soil quality score (0-1).
        water_quality (float): Water quality score (0-1).
        invasive_species (float): Invasive species presence score (0-1).
        fauna_diversity (float): Fauna diversity score (0-1).
    Returns:
        float: Calculated habitat condition score.
    '''

    return (
        vegetation_cover * 0.25 +
        soil_quality * 0.25 +
        water_quality * 0.2 +
        (1 - invasive_species) * 0.15 +
        fauna_diversity * 0.15
    )

