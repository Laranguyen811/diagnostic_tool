from typing import Union, Iterable, List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import warnings

from numpy import ndarray

from utils.validation import validate_range,validate_array_values, validate_type_schema
import math
import warnings
import gower
from scipy.spatial.distance import pdist, squareform
from skbio.stats.ordination import pcoa
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler, OneHotEncoder
def calculate_biodiversity_units(unit_data: List[Dict[str,Any]]) -> float:
    '''
    Calculates the biodiversity units based on area, distinctiveness, condition, strategic significance, and connectivity.
    Args:
        unit_data (List[Dict[str, Any]]): A list of dictionaries of biodiversity units, including area, distinctiveness, condition,strategic_significance, and connectivity
    Returns:
        float: Calculated biodiversity units.
    Example:
        >>> unit_data = [{
        ...     "area": 10.0,
        ...     "distinctiveness": 0.8,
        ...     "condition": 0.9,
        ...     "strategic_significance": 1.0,
        ...     "connectivity": 0.7,
        ... }]
        >>> calculate_biodiversity_units(unit_data)
        5.04
    '''
    required_keys = {
    "area": (0.01, float("inf")),
    "distinctiveness": (0.0, 1.0),
    "condition": (0.0, 1.0),
    "strategic_significance": (0.0, 1.0),
    "connectivity": (0.0, 1.0),
}
    type_schema = {
        "area": (float,float),
        "distinctiveness": (float,float),
        "condition":(float,float),
        "strategic_significance":(float,float),
        "connectivity":(float,float)
    }

    validate_type_schema(unit_data,type_schema)
    for record in unit_data:
        validate_range(record["area"],0.01,float("inf"),"area")
        for key in ["distinctiveness", "condition","strategic_significance", "connectivity"]:
            validate_range(record[key],0.0,1.0,key)
        return (
                  record["area"]
                * record["distinctiveness"]
                * record["condition"]
                * record["strategic_significance"]
                * record["connectivity"]
    )

def calculate_species_richness(
        specimen_richness_data: List[Dict[str,Any]],
        strict: bool = True,
) -> Optional[Union[int, float]]:
    '''
    Calculates the species richness based on the total number of species and the area.
    Args:
        total_species (int): Total number of species in the area.
        area (float): Area of the habitat in hectares.
    Returns:
        Union[int, float]: Calculated species richness (species per hectare).
    Example:
        >>> specimen_richness_data = [{
        ...     "total_species": 100,
        ...     "area": 50.0,
        ... }]
        >>> calculate_species_richness(specimen_richness_data)
        2.0
    '''
    required_keys = {
        "total_species": (0,float("inf")),
        "area": (0.0,float("inf")),
    }
    for record in specimen_richness_data:
        total_species = record["total_species"]
        area = record ["area"]
        for key in ["total_species","area"]:
            validate_range(record[key],0,float("inf"),key)
        if area == 0:
            if strict:
                raise ValueError("Area must be greater than zero for strict mode.")
            else:
            # In non-strict mode, return NaN or zero to indicate invalid calculation
                return float('nan')

        return total_species / area
    return None

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
    condition_data: List[Dict[str,Any]],
) -> Optional[float]:
    '''
    Calculates the habitat condition score based on vegetation cover, soil quality, water quality, invasive species presence, and fauna diversity.
    Args:
        condition_data (dict): A dictionary of habitat condition parameters, including vegetation_cover, soil_quality, water_quality, invasive_species, and fauna_diversity
    Returns:
        float: Calculated habitat condition score.
    Example:
        >>> condition_data = [{
        ...     "vegetation_cover": 80.0,
        ...     "soil_quality": 0.9,
        ...     "water_quality": 0.8,
        ...     "invasive_species": 0.1,
        ...     "fauna_diversity": 0.7,
        ... }]
        >>> calculate_habitat_condition_score(condition_data)
        0.4032
    '''
    required_keys ={
        "vegetation_cover": (0.0, 100.0),
        "soil_quality": (0.0, 1.0),
        "water_quality": (0.0, 1.0),
        "invasive_species": (0.0, 1.0),
        "fauna_diversity": (0.0, 1.0),
    }
    type_schema = {
        "vegetation_cover":(float,float),
        "soil_quality":(float,float),
        "water_quality":(float,float),
        "invasive_species":(float,float),
        "fauna_diversity":(float,float)
    }
    validate_type_schema(condition_data,type_schema)
    for record in condition_data:
        for key, (min_val, max_val) in required_keys.items():
            validate_range(record[key], min_val, max_val, name=key)
        score = (record["vegetation_cover"] / 100.0) * record["soil_quality"] * record["water_quality"] * (1 - record["invasive_species"]) * record["fauna_diversity"]
        if not math.isfinite(score):
            raise ValueError("Habitat condition score must be finite.")
        return score
    return None

def calculate_endemism_index(
    endemism_data: List[Dict[str, Any]]
) -> Tuple[float, int]:
    """
    Calculates the endemism index based on the number of endemic species and total species.

    Each record must include:
        - 'presence_or_absence': 1 if species is present, 0 if absent
        - 'total_regions': int > 0

    Args:
        endemism_data (List[Dict[str, Any]]): List of dictionaries of endemism parameters

    Returns:
        Tuple[float, int]: Calculated endemism index and number of skipped records
    """
    required_keys = {
        "presence_or_absence": (0, 1),
        "total_regions": (1, float("inf")),
    }

    weighted_endemic_index = 0.0
    num_skipped = 0

    for record in endemism_data:
        try:
            for key, (min_val, max_val) in required_keys.items():
                validate_range(record[key], min_val, max_val, key)
            if record["presence_or_absence"] == 1:
                weighted_endemic_index += 1 / record["total_regions"]
        except (ValueError, TypeError) as e:
            warnings.warn(f"Skipping invalid record: {e}")
            num_skipped += 1
            continue

    return weighted_endemic_index, num_skipped

def build_required_keys(
        trait_fields:List[str],
        id_type: Tuple[type, ...] = (str, float),
        abundance_type: Tuple[type, ...]=(float,),
        trait_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Schema with type and optional range info.
.
    Inputs:
        trait_fields (List[str]): List of trait fields.
        id_type (Tuple[type]): Type of species ID.
        abundance_type (Tuple[type, ...]): Accepted types for abundance.
        trait_range(Optional[Dict[str, Tuple[float, float]]]): a range that a specified trait needs to be bounded by.

    Returns:
        Dict: Dictionary of required keys.
    """
    keys = {
        "species_id":{"types":id_type},
        "abundance": {"types":abundance_type}
    }
    for trait in trait_fields:
        keys[trait] = {"types":(str,float)}
        if trait_ranges and trait in trait_ranges:
            keys[trait]["range"] = trait_ranges[trait]
    return keys

def prepare_trait_matrix(trait_data: List[Dict[str,Any]], trait_count: int = 6) -> ndarray:
    """
    Prepare trait matrix for functional diversity calculation.
    Args:
        trait_data (List[Dict[str,Any]]): A dictionary of trait parameters
        trait_count (int, optional): Number of trait parameters. Defaults to 6.
    Returns:
        ndarray: Trait matrix.
    """
    trait_fields = [f"trait_{i}" for i in range(1, trait_count +1)]
    trait_ranges = {
    trait:(float("-inf"), float("inf")) for trait in trait_fields
    }
    required_keys = build_required_keys(trait_fields,trait_ranges=trait_ranges)
    type_schema = {
        key: meta["types"]
        for key, meta in required_keys.items()
    }
    validate_type_schema(trait_data, type_schema)
    oh = OneHotEncoder()
    std_scaler = StandardScaler()
    df = pd.DataFrame(trait_data)
    trait_df = df[trait_fields]
    cat_cols = [] # Categorical columns
    cont_cols = [] # Continuous columns
    for key in trait_df.columns:
        if trait_df[key].dtype == 'object' or trait_df[key].apply(lambda x: isinstance(x,str)).any():
            cat_cols.append(key)
        else:
            cont_cols.append(key)

    encoded = oh.fit_transform(trait_df[cat_cols]) if cat_cols else np.empty((len(trait_df), 0))
    scaled = std_scaler.fit_transform(trait_df[cont_cols]) if cont_cols else np.empty((len(trait_df), 0))
    trait_matrix = np.hstack([encoded,scaled]) # Horizontally stack encoded and scaled columns
    return trait_matrix

def calculate_functional_richness(
        trait_data: List[Dict[str,Any]],
        trait_count: int=6,
)-> float:
    """
    Calculate functional diversity, categorising based on traits of species.
    Inputs:
        trait_data (List[Dict[str,Any]]): A dictionary of trait parameters
    Returns:
        float: Calculated functional richness.
    """
    trait_matrix = prepare_trait_matrix(trait_data,trait_count)
    if trait_matrix.shape[0] <= trait_matrix.shape[1]:
        return 0.0
    hull = ConvexHull(trait_matrix, incremental=True)
    functional_richness = hull.volume
    return functional_richness
