from typing import Union, Iterable
import numpy as np
import pandas as pd

def validate_range(
        value: Union[float,int],
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
    Raises:
        Value Error: If the value is outside a specific range
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
    Raises:
        ValueError: If any value is outside the specified range.
    """
    values = np.asarray(values)
    if not np.all(min_val <= values) & np.all(values <= max_val):
        invalid = values[(values < min_val) | (values > max_val)]
        raise ValueError(f"{name} contains values outside[{min_val}, {max_val}]: {invalid.tolist()}")

def validate_input_dict(
        inputs: dict,
        schema: dict
) -> None:
    """
    Validates a dictionary of inputs against a schema of an expected range.
    Args:
        inputs(dict): Dictionary of inputs
        schema(dict): Dictionary of expected ranges
    Raises:
        ValueError: If any input is missing or out of range.
    """
    for key, (min_val, max_val) in schema.items():
        if key not in inputs:
            raise ValueError(f"Missing input: {key}")
        validate_range(inputs[key], min_val, max_val, name=key)
