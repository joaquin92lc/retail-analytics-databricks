
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock
from src.mlflow_quality_gate import validate_mlflow_run_by_id
from src.mlflow_quality_gate import evaluate_mlflow_run
from src.model_quality_gate import ModelQualityError

from src.mlflow_quality_gate import validate_mlflow_run


# Simular una ejecución de MLflow.
def create_mock_run(wape=None, rmse=None):
    return SimpleNamespace(
        data=SimpleNamespace(
            metrics={
                "test_wape": wape,
                "test_rmse": rmse,
            }
        )
    )


# TEST 1: El modelo cumple los límites.
def test_valid_mlflow_run():
    run = create_mock_run(wape=15.0, rmse=2000.0)

    assert validate_mlflow_run(run) is True


# TEST 2: El WAPE supera el límite.
def test_mlflow_run_exceeds_wape():
    run = create_mock_run(wape=25.0, rmse=2000.0)

    with pytest.raises(ValueError, match="WAPE"):
        validate_mlflow_run(run)


# TEST 3: Falta una métrica obligatoria.
def test_mlflow_run_missing_metric():
    run = create_mock_run(wape=15.0, rmse=None)

    with pytest.raises(ValueError, match="Faltan"):
        validate_mlflow_run(run)


# TEST 4: Recuperar y validar una ejecución de MLflow.
def test_validate_mlflow_run_by_id():

    client = MagicMock()

    client.get_run.return_value = create_mock_run(
        wape=15.0,
        rmse=2000.0
    )

    result = validate_mlflow_run_by_id(
        "run_123",
        client
    )

    assert result is True
    client.get_run.assert_called_once_with("run_123")


# TEST 5: Rechazar un Run ID vacío.
def test_validate_mlflow_run_by_id_invalid():

    client = MagicMock()

    with pytest.raises(ValueError, match="Run ID"):
        validate_mlflow_run_by_id("", client)

    client.get_run.assert_not_called()


# TEST 6: Modelo aprobado.
def test_evaluate_mlflow_passed():
    client = MagicMock()
    client.get_run.return_value = create_mock_run(15.0, 2000.0)

    result = evaluate_mlflow_run("run_123", client)

    assert result["status"] == "PASSED"


# TEST 7: Modelo rechazado por WAPE.
def test_evaluate_mlflow_rejected():
    client = MagicMock()
    client.get_run.return_value = create_mock_run(25.0, 2000.0)

    result = evaluate_mlflow_run("run_123", client)

    assert result["status"] == "REJECTED"


# TEST 8: MLflow no disponible.
def test_evaluate_mlflow_connection_error():
    client = MagicMock()
    client.get_run.side_effect = ConnectionError(
        "MLflow no disponible"
    )

    result = evaluate_mlflow_run("run_123", client)

    assert result["status"] == "ERROR"
    assert "MLflow no disponible" in result["reason"]

# TEST 9: La ejecución solicitada no existe en MLflow.
def test_evaluate_mlflow_run_not_found():
    client = MagicMock()

    client.get_run.side_effect = Exception(
        "Run ID no encontrado"
    )

    result = evaluate_mlflow_run("run_inexistente", client)

    assert result["status"] == "ERROR"
    assert "Run ID no encontrado" in result["reason"]

    # TEST 10: Clasificar por tipo de excepción, no por mensaje.
def test_rejection_independent_of_error_message():
    client = MagicMock()

    client.get_run.side_effect = ModelQualityError(
        "El modelo no cumple la política de calidad"
    )

    result = evaluate_mlflow_run("run_123", client)

    assert result["status"] == "REJECTED"