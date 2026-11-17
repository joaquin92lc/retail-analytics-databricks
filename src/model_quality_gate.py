
import math


# ============================================================
# 1. UMBRALES DE CALIDAD DEL MODELO
# ============================================================

MAX_WAPE = 20.0
MAX_RMSE = 2500.0


# ============================================================
# 2. EXCEPCIÓN PERSONALIZADA
# ============================================================

class ModelQualityError(ValueError):
    """
    Excepción que indica que el modelo incumple
    los umbrales de calidad establecidos.
    """
    pass


# ============================================================
# 3. VALIDACIÓN DEL MODELO
# ============================================================

def validate_model(wape, rmse):
    """
    Comprueba si las métricas del modelo cumplen
    los requisitos de calidad.

    Args:
        wape (float): Error porcentual ponderado.
        rmse (float): Raíz del error cuadrático medio.

    Returns:
        bool: True si el modelo supera el Quality Gate.

    Raises:
        ValueError: Métricas ausentes o inválidas.
        ModelQualityError: Incumplimiento de los umbrales.
    """

    # 1. Verificar métricas obligatorias.
    if wape is None or rmse is None:
        raise ValueError("Faltan métricas obligatorias")

    # 2. Validar los valores recibidos.
    for name, value in {"WAPE": wape, "RMSE": rmse}.items():

        # Comprobar que sean valores numéricos.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} debe ser numérico")

        # Rechazar valores NaN o infinitos.
        if not math.isfinite(value):
            raise ValueError(f"{name} no es finito")

        # Rechazar valores negativos.
        if value < 0:
            raise ValueError(f"{name} no puede ser negativo")

    # 3. Comprobar los límites permitidos.
    if wape > MAX_WAPE:
        raise ModelQualityError(
            f"WAPE supera el límite: {wape}%"
        )

    if rmse > MAX_RMSE:
        raise ModelQualityError(
            f"RMSE supera el límite: {rmse}"
        )

    # 4. Modelo aprobado.
    return True