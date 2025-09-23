from typing import Union, Iterable, List, Dict, Any, Tuple
import numpy as np
import pandas as pd

def validate_range(
        value: Union[Iterable[Union[int, float]], np.ndarray, pd.Series],
        min_val: float,
        max_val: float,
        name: str = "value"
) -> None:
    """
    Validates a scalar value is within a specific range.
    Args:
        value(float or int): Value to validate
        min_val(float): Mininum acceptable value
        max_val(float): Maximum acceptable value
        name (str): Name of the value to be used in error messages
    Raises:
        Value Error: If the value is outside a specified range
        :rtype: None
    """
    if not min_val <= value <= max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")

def validate_array_values(
            values: Union[Iterable[float], np.ndarray, pd.Series],
            min_val: float,
            max_val: float,
            name: str = "array"
    ) -> None:
    """
    Validate all values in an array-like structure are within a specific range.
    Args:
        values(Iterable): Array-like structure of numeric values
        min_val(float): Minimum acceptable value
        max_val(float): Maximum acceptable value
        name (str): Name of the array to be used in error messages
    Raises:
        ValueError: If any value is outside the specified range.
    """
    values = np.asarray(values)
    if not np.all(min_val <= values) and np.all(values <= max_val):
        invalid = values[(values < min_val) | (values > max_val)]
        raise ValueError(f"{name} contains values outside[{min_val}, {max_val}]: {invalid.tolist()}")

def validate_type_schema(
        inputs: List[Dict[str,Any]],
        type_schema: Dict[str,Tuple[type,...]]
) -> None:
    """
    Validates a dictionary of inputs against a schema of an expected range.
    Args:
        inputs(List[Dict[str,Any]]: A list of dictionaries containing input data
        type_schema(Dict[str, Tuple[type,...]]): Dictionary of expected ranges
    Raises:
        TypeError: If any input is not a float or int.
    """
    for i, record in enumerate(inputs):
        for key, expected_type in type_schema.items():
            if key not in record:
                raise ValueError(f"Record {i} is missing input: {key}")
            if not isinstance(record[key], expected_type):
                raise TypeError(f"Record {i}, key {key} has type {type(record[key]).__name__}"
                                f"expected one of: {[t.__name__ for t in expected_type]}"
                                )
