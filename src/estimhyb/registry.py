"""Registro de estimadores y de escenarios del estudio.

Un solo lugar define qué se compara y bajo qué condiciones. Cualquier cambio
aquí queda registrado en el control de versiones y afecta a todas las corridas
por igual, lo que evita que un comparador reciba un trato distinto de otro.
"""
import json
import os

from . import augment
from .filters import adapters
from .sensors import CHANNELS, ChannelDegradation

_CFG = os.path.join(os.path.dirname(__file__), "..", "..", "configs",
                    "tuning.json")


def load_tuning():
    with open(_CFG, encoding="utf-8") as fh:
        return json.load(fh)


# Escala de Q del comparador mal sintonizado. Subestima el ruido de proceso en
# un orden de magnitud, lo que produce un filtro sobreconfiado. La elección es
# declarada y no se ajusta según los resultados.
MISTUNED_FACTOR = 0.1

# Severidades del barrido. Cero corresponde al escenario sin degradación.
SEVERITIES = (0.0, 0.25, 0.5, 0.75, 1.0)

# Canal degradado por defecto en cada modo. El GNSS es el canal cuya pérdida
# de fiabilidad tiene mayor efecto sobre el error de posición, y el
# magnetómetro es el canal cuyo sesgo se propaga al rumbo.
SCENARIOS = {
    "S0_nominal": {},
    "S1_noise": {"gnss": "noise"},
    "S2_bias": {"mag": "bias"},
    "S3_outlier": {"gnss": "outlier"},
    "S4_dropout": {"gnss": "dropout"},
    "S5_combined": {"gnss": "combined", "mag": "bias"},
}


def degradation_for(scenario, severity, onset=20.0):
    """Construye la configuración de degradación de un escenario y severidad."""
    if scenario not in SCENARIOS:
        raise ValueError("escenario desconocido, %s" % scenario)
    spec = SCENARIOS[scenario]
    out = {}
    for channel, mode in spec.items():
        if channel not in CHANNELS:
            raise ValueError("canal desconocido, %s" % channel)
        out[channel] = ChannelDegradation(mode, severity, onset)
    return out


def estimators(tuning=None):
    """Los seis comparadores del cuerpo del artículo, más la ablación.

    Devuelve un diccionario ordenado de etiqueta a par formado por la fábrica
    del adaptador y la escala de Q. La ablación se reporta solo en la figura
    de ablación, no en la tabla principal.
    """
    tuning = tuning or load_tuning()
    q_nom = float(tuning["q_scale_nominal"])
    return {
        "EKF_nom": (lambda n: adapters.NoiseAdapter(n), q_nom),
        "EKF_mis": (lambda n: adapters.NoiseAdapter(n), q_nom * MISTUNED_FACTOR),
        "CovMatch": (lambda n: adapters.CovarianceMatchingAdapter(n), q_nom),
        "SageHusa": (lambda n: adapters.SageHusaAdapter(n), q_nom),
        "Huber": (lambda n: adapters.HuberAdapter(n), q_nom),
        "Propuesto": (lambda n: adapters.DwellTimeAdapter(n), q_nom),
    }


def compensating(tuning=None):
    """Brazo con compensacion de sesgo disparada. Ver src/estimhyb/augment.py.

    Requiere el filtro aumentado, de modo que se entrega con la clase que le
    corresponde. El resto del pipeline no cambia.
    """
    tuning = tuning or load_tuning()
    q_nom = float(tuning["q_scale_nominal"])
    return {
        "Compensado": (lambda n: adapters.DwellTimeAdapter(n, compensate=True),
                       q_nom, augment.AugmentedBatchEKF),
        # Compensacion siempre activa, sin disparo. Aisla lo que aporta
        # activar solo ante evidencia en lugar de llevar el termino siempre.
        "Comp_siempre": (
            lambda n: adapters.DwellTimeAdapter(n, compensate=True,
                                                alpha_bias=1.0),
            q_nom, augment.AugmentedBatchEKF),
    }


def ablations(tuning=None):
    tuning = tuning or load_tuning()
    q_nom = float(tuning["q_scale_nominal"])
    return {
        # Solo inflacion acotada. Elemento uno del esquema.
        "Abl_solo_inflacion": (
            lambda n: adapters.ablation_adapter(n), q_nom),
        # Sin la prueba del primer momento. Aisla lo que aporta detectar sesgo.
        "Abl_sin_media": (
            lambda n: adapters.DwellTimeAdapter(n, mean_test=False), q_nom),
        # Sin zona muerta. Aisla lo que aporta no reaccionar a fluctuacion.
        "Abl_sin_zona": (
            lambda n: adapters.DwellTimeAdapter(n, dead_zone=0.0), q_nom),
        # Sin tiempo minimo de permanencia. Aisla lo que aporta acotar la
        # frecuencia de conmutacion entre excluir y readmitir.
        "Abl_sin_permanencia": (
            lambda n: adapters.DwellTimeAdapter(n, dwell_up=1, dwell_hold=0),
            q_nom),
    }
