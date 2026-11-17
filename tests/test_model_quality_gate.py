
import pytest

from src.model_quality_gate import validate_model


# Test 1: modelo que cumple los requisitos
def test_valid_model():
    assert validate_model(15, 2000) is True


# Test 2: WAPE superior al límite
def test_wape_exceeds_limit():
    with pytest.raises(ValueError):
        validate_model(32, 2000)


# Test 3: RMSE superior al límite
def test_rmse_exceeds_limit():
    with pytest.raises(ValueError):
        validate_model(15, 3000)


# Test 4: métrica obligatoria ausente
def test_missing_metric():
    with pytest.raises(ValueError):
        validate_model(15, None)


# Test 5: valor NaN no permitido
def test_nan_metric():
    with pytest.raises(ValueError):
        validate_model(float("nan"), 2000)
