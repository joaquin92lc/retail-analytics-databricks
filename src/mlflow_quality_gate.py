
from src.model_quality_gate import (
    validate_model,
    ModelQualityError,
)


# ============================================================
# 1. VALIDAR LAS MÉTRICAS DE UNA EJECUCIÓN MLFLOW
# ============================================================

def validate_mlflow_run(run):
    """
    Extrae las métricas de test de una ejecución
    MLflow y las valida con el Quality Gate.
    """

    metrics = run.data.metrics

    wape = metrics.get("test_wape")
    rmse = metrics.get("test_rmse")

    return validate_model(wape, rmse)


# ============================================================
# 2. RECUPERAR UNA EJECUCIÓN MEDIANTE RUN ID
# ============================================================

def validate_mlflow_run_by_id(run_id, client):
    """
    Recupera una ejecución existente mediante
    un cliente MLflow y valida sus métricas.
    """

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("El Run ID debe ser un texto no vacío")

    run = client.get_run(run_id)

    return validate_mlflow_run(run)


# ============================================================
# 3. EVALUAR Y CLASIFICAR EL RESULTADO
# ============================================================

def evaluate_mlflow_run(run_id, client):
    """
    Evalúa una ejecución y devuelve un estado
    estructurado para su uso en pipelines MLOps.

    PASSED: modelo aprobado.
    REJECTED: incumplimiento de calidad.
    ERROR: evaluación no completada correctamente.
    """

    try:
        validate_mlflow_run_by_id(run_id, client)

        return {
            "status": "PASSED",
            "reason": "El modelo cumple los criterios de calidad",
        }

    except ModelQualityError as error:
        return {
            "status": "REJECTED",
            "reason": str(error),
        }

    except ValueError as error:
        return {
            "status": "ERROR",
            "reason": str(error),
        }

    except Exception as error:
        return {
            "status": "ERROR",
            "reason": (
                f"Error al recuperar o validar MLflow: {error}"
            ),
        }
