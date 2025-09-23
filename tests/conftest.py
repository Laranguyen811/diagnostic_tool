import pytest
import logging
@pytest.fixture
def sample_input():
    return {"presence_or_absence": 1, "total_regions": 10}

def python_collection_modifyitems(items):
    for item in items:
        item.name = item.name.upper()


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: mark as unit test")

@pytest.mark.unit
def test_unit_function():
    assert 1 == 1

logging.basicConfig(level=logging.DEBUG,
                    format="%(levelname)s:%(name)s:%(message)s")