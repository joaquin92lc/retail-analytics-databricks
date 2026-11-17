
import math

MAX_WAPE = 20.0
MAX_RMSE = 2500.0


def validate_model(wape, rmse):
    """Comprueba si las métricas cumplen los requisitos."""

    # 1. Verificar métricas obligatorias
    if wape is None or rmse is None:
        raise ValueError("Faltan métricas obligatorias")

    # 2. Validar los valores recibidos
    for name, value in {"WAPE": wape, "RMSE": rmse}.items():

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} debe ser numérico")

        if not math.isfinite(value):
            raise ValueError(f"{name} no es finito")

        if value < 0:
            raise ValueError(f"{name} no puede ser negativo")

    # 3. Comprobar los límites permitidos
    if wape > MAX_WAPE:
        raise ValueError(f"WAPE supera el límite: {wape}%")

    if rmse > MAX_RMSE:
        raise ValueError(f"RMSE supera el límite: {rmse}")

    return True
