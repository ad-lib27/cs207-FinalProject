import numpy as np
from itertools import product as prod
import math
import pytest

from adlib27.autodiff import AutoDiff as AD
from adlib27.elem_function import sin
from adlib27.optimize import optimize

#test optimize
def test_optimize():
    assert optimize([-1.0, 0.0, 1.0], ['x'], '3*x') == {'global maximum': {'input range': ([1.0], [1.0]), 'value range': (3.0, 3.0), 'inflection type': 'endpoint'}, 'global minimum': {'input range': ([-1.0], [-1.0]), 'value range': (-3.0, -3.0), 'inflection type': 'endpoint'}, 'all critical points': [{'variables': ['x'], 'input range': ([-1.0], [-1.0]), 'value range': (-3.0, -3.0), 'inflection type': 'endpoint'}, {'input range': ([1.0], [1.0]), 'value range': (3.0, 3.0), 'inflection type': 'endpoint'}]}
    assert optimize([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['a', 'b'], '3*a*sin(b)') == {'global maximum': {'input range': ([1.0, 1.0], [1.0, 1.0]), 'value range': (2.5244129544236893, 2.5244129544236893), 'inflection type': 'endpoint'}, 'global minimum': {'input range': ([0.0, 0.0], [0.0, 0.0]), 'value range': (0.0, 0.0), 'inflection type': 'endpoint'}, 'all critical points': [{'variables': ['a', 'b'], 'input range': ([0.0, 0.0], [0.0, 0.0]), 'value range': (0.0, 0.0), 'inflection type': 'endpoint'}, {'input range': ([1.0, 1.0], [1.0, 1.0]), 'value range': (2.5244129544236893, 2.5244129544236893), 'inflection type': 'endpoint'}, {'variables': ['a', 'b'], 'input range': ([1.0, 1.0], [1.2, 1.2]), 'value range': (2.5244129544236893, 0.0), 'inflection type': 'critical point'}, {'variables': ['a', 'b'], 'input range': ([1.0, 1.0], [1.2, 1.2]), 'value range': (2.5244129544236893, 0.0), 'inflection type': 'critical point'}, {'variables': ['a', 'b'], 'input range': ([1.0, 1.0], [1.2, 1.2]), 'value range': (2.5244129544236893, 0.0), 'inflection type': 'critical point'}]}