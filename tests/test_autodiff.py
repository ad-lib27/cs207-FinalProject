import pytest
from adlib27.autodiff import AutoDiff as AD
import numpy as np

# Testing the getters and setters
def test_getters():
    x = AD(val=[10])
    value = x.val
    derivative = x.der
    assert value == pytest.approx([10])
    for d in derivative:
        assert d == pytest.approx([1])

def test_setters():
    x = AD(val=[10])
    x.val[0] = 5
    x.der[0][0] = 0
    assert x.val == pytest.approx([5])
    for d in x.der:
        assert d == pytest.approx([0])

# Testing the Unary operations
def test_neg():
    x = AD(val=[10])
    y = -x
    assert y.val == pytest.approx([-10])
    for d in y.der:
        assert d == pytest.approx([-1])

def test_neg_many_var():
    x = AD(val=[10], index=0, magnitude=2)
    y = AD(val=[20], index=1, magnitude=2)
    z = -(x + y)
    assert z.val == pytest.approx([-30])
    for d in z.der:
        assert d == pytest.approx([-1])

def test_neg_many_val():
    x = AD(val=[10, 1, 3, 4])
    y = -x
    assert y.val == pytest.approx([-10, -1, -3, -4])
    for d in y.der:
        assert d == pytest.approx([-1, -1, -1, -1])

def test_pos():
    x = AD(val=[-15])
    y = +x
    assert y.val == pytest.approx([-15])
    for d in y.der:
        assert d == pytest.approx([1])

# Testing the basic operations (+, -, *, /)
# Testing the add and radd
def test_add_const():
    x = AD(val=[10])
    y = x + 22
    assert y.val == pytest.approx([32])
    for d in y.der:
        assert d == pytest.approx([1])

def test_add_AD():
    x = AD(val=[3])
    y = x + x
    assert y.val == pytest.approx([6])
    for d in y.der:
        assert d == pytest.approx([2])

def test_radd_const():
    x = AD(val=[10])
    y = 5 + x
    assert y.val == pytest.approx([15])
    for d in y.der:
        assert d == pytest.approx([1])

# Testing the sub and rsub
def test_sub_const():
    x = AD(val=[10])
    y = x - 3
    assert y.val == pytest.approx([7])
    for d in y.der:
        assert d == pytest.approx([1])

def test_sub_AD():
    x1 = AD(val=[14])
    x2 = AD(val=[3])
    y = x1 - x2
    assert y.val == pytest.approx([11])
    for d in y.der:
        assert d == pytest.approx([0])

def test_rsub_const():
    x = AD(val=[1])
    y = 7 - x
    assert y.val == pytest.approx([6])
    for d in y.der:
        assert d == pytest.approx([-1])

# Testing the mul and rmul
def test_mul_const():
    x = AD(val=[10])
    y = x*3
    assert y.val == pytest.approx([30])
    for d in y.der:
        assert d == pytest.approx([3])

def test_mul_AD():
    x1 = AD(val=[5])
    x2 = AD(val=[3])
    y = x1 * x2
    assert y.val == pytest.approx([15])
    for d in y.der:
        assert d == pytest.approx([8])

def test_rmul_const():
    x = AD(val=[6])
    y = 2*x
    assert y.val == pytest.approx([12])
    for d in y.der:
        assert d == pytest.approx([2])

# Testing the div and rdiv
def test_div_const():
    x = AD(val=[20])
    y = x/4
    assert y.val == pytest.approx([5])
    for d in y.der:
        assert d == pytest.approx([0.25])

def test_div_AD():
    x1 = AD(val=[15])
    x2 = AD(val=[5])
    y = x1 / x2
    assert y.val == pytest.approx([3])
    for d in y.der:
        assert d == pytest.approx([-0.4])

def test_rdiv_const():
    x = AD(val=[5])
    y = 10/x
    assert y.val == pytest.approx([2])
    for d in y.der:
        assert d == pytest.approx([-0.4])

# Testing the power and rpower
def test_power_const():
    x = AD(val=[5])
    y = x**2
    assert y.val == pytest.approx([25])
    for d in y.der:
        assert d == pytest.approx([10])

def test_power_AD():
    x1 = AD(val=[1])
    x2 = AD(val=[3])
    y = x1 ** x2
    assert y.val == pytest.approx([1])
    for d in y.der:
        assert d == pytest.approx([3])

def test_rpower_const():
    x = AD(val=[2])
    y = 10**x
    assert y.val == pytest.approx([100])
    for d in y.der:
        assert d == pytest.approx([10**2* np.log(10)])
