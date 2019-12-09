import numpy as np
import math
import pytest

from adlib27.autodiff import AutoDiff as AD
from adlib27.elem_function import sin
from adlib27.optimize import optimize

def test_optimize():
    vals = np.linspace(0, 2 * math.pi, 650)
    x = AD(val=vals)
    r = optimize(sin(x), vals)
    assert r["global maximum"]["input range"] == pytest.approx((1.568375993471638, 1.5780573267646727))
    assert r["global minimum"]["input range"] == pytest.approx((4.705127980414914, 4.714809313707948))
    assert len(r["all extrema"]) == 4
