import math
from typing import Tuple


def angle_of_vector(vector: Tuple[float, float]) -> float:
    """
    Returns the angle of a vector in radians...
    :param vector:
    :return:
    """
    unit = into_unit(vector)

    dot = unit[0] * 1 + unit[1] * 0
    det = unit[0] * 0 - unit[1] * 1

    angle = -math.atan2(det, dot) + math.pi
    return angle

def vector_norm(vector: Tuple[float, float]) -> float:
    """
    Compute the norm of a 2D vector
    :param vector:
    :return:
    """
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2)


def into_unit(vector: Tuple[float, float]) -> Tuple[float, float]:
    """
    Turn a vector into a unit vector
    :param vector:
    :return:
    """
    norm = vector_norm(vector)
    return (vector[0] / norm, vector[1] / norm)
