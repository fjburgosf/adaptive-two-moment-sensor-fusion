"""Utilidades geométricas compartidas."""
import numpy as np


def wrap_pi(angle):
    """Envuelve un ángulo, o un arreglo de ángulos, al intervalo (-pi, pi]."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi
