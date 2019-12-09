import pytest
from adlib27.autodiff import AutoDiff as AD
import numpy as np

# Testing the getters and setters

def test_getters():
    x = AD(val=[10])
    value = x.val
    derivative = x.der
    assert value == pytest.approx([10], rel=1e-4)
    for d in derivative:
        assert d == pytest.approx([1])

def test_setters():
    x = AD(val=[10])
    x.val[0] = 5
    x.der[0][0] = 0
    assert x.val == pytest.approx([5])
    for d in x.der:
        assert d == pytest.approx([0])

# Testing the comparison operations

def test_ne_const():
    x = AD(val=[10])
    y = 10
    assert x != y

def test_ne_different_AD():
    x = AD(val=[10], index=0, magnitude=2)
    y = AD(val=[20], index=1, magnitude=2)
    assert x != y

def test_eq_AD():
    x1 = AD(val=[10])
    x2 = AD(val=[10])
    assert x1 == x2

def test_ne_different_AD_many_val():
    x = AD(val=[10, -1, 3.2, 4], index=0, magnitude=2)
    y = AD(val=[-2, 0, 1, 100], index=1, magnitude=2)
    assert x != y

def test_eq_AD_many_val():
    x1 = AD(val=[10, 1, 3, 4])
    x2 = AD(val=[10, 1, 3, 4])
    assert x1 == x2

# Testing the Unary operations

def test_neg():
    x = AD(val=[10])
    y = -x
    assert y.val == pytest.approx([-10])
    for d in y.der:
        assert d == pytest.approx([-1])

def test_neg_different_AD():
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

def test_neg_different_AD_many_val():
    x = AD(val=[10, -1, 3.2, 4], index=0, magnitude=2)
    y = AD(val=[-2, 0, 1, 100], index=1, magnitude=2)
    z = -(x + y)
    assert z.val == pytest.approx([-8, 1, -4.2, -104])
    for d in z.der:
        assert d == pytest.approx([-1, -1, -1, -1])

def test_pos():
    x = AD(val=[-15])
    y = +x
    assert y.val == pytest.approx([-15])
    for d in y.der:
        assert d == pytest.approx([1])

def test_pos_different_AD():
    x = AD(val=[10], index=0, magnitude=2)
    y = AD(val=[-20], index=1, magnitude=2)
    z = +(x + y)
    assert z.val == pytest.approx([-10])
    for d in z.der:
        assert d == pytest.approx([1])

def test_pos_many_val():
    x = AD(val=[10, 1, 3, 4])
    y = +x
    assert y.val == pytest.approx([10, 1, 3, 4])
    for d in y.der:
        assert d == pytest.approx([1, 1, 1, 1])

def test_pos_different_AD_many_val():
    x = AD(val=[10, -1, 3.2, 4], index=0, magnitude=2)
    y = AD(val=[-2, 0, 1, 100], index=1, magnitude=2)
    z = +(x + y)
    assert z.val == pytest.approx([8, -1, 4.2, 104])
    for d in z.der:
        assert d == pytest.approx([1, 1, 1, 1])

# Testing the basic operations (+, -, *, /)

# Testing the add and radd
def test_add_const():
    x = AD(val=[10])
    y = x + 22
    assert y.val == pytest.approx([32])
    for d in y.der:
        assert d == pytest.approx([1])

def test_add_const_many_val():
    x = AD(val=[-1, 0, 5, 10])
    y = x + 17
    assert y.val == pytest.approx([16, 17, 22, 27])
    for d in y.der:
        assert d == pytest.approx([1, 1, 1, 1])

def test_add_AD():
    x = AD(val=[3])
    y = x + x
    assert y.val == pytest.approx([6])
    for d in y.der:
        assert d == pytest.approx([2])

def test_add_AD_many_val():
    x = AD(val=[-1, 0, 5, 10])
    y = x + x
    assert y.val == pytest.approx([-2, 0, 10, 20])
    for d in y.der:
        assert d == pytest.approx([2, 2, 2, 2])

def test_add_different_AD():
    x = AD(val=[3], index=0, magnitude=2)
    y = AD(val=[4], index=1, magnitude=2)
    z = x + y
    assert z.val == pytest.approx([7])
    for d in z.der:
        assert d == pytest.approx([1])

def test_add_different_AD_many_val():
    x = AD(val=[10, -1, 3.2, 4], index=0, magnitude=2)
    y = AD(val=[-2, 0, 1, 100], index=1, magnitude=2)
    z = x + y
    assert z.val == pytest.approx([8, -1, 4.2, 104])
    for d in z.der:
        assert d == pytest.approx([1, 1, 1, 1])

def test_radd_const():
    x = AD(val=[10])
    y = 5 + x
    assert y.val == pytest.approx([15])
    for d in y.der:
        assert d == pytest.approx([1])

def test_radd_const_many_val():
    x = AD(val=[-1, 0, 5, 10])
    y = 17 + x
    assert y.val == pytest.approx([16, 17, 22, 27])
    for d in y.der:
        assert d == pytest.approx([1, 1, 1, 1])

# Testing the sub and rsub
def test_sub_const():
    x = AD(val=[10])
    y = x - 3
    assert y.val == pytest.approx([7])
    for d in y.der:
        assert d == pytest.approx([1])

def test_sub_const_many_val():
    x = AD(val=[-1, 0, 5, 10])
    y = x - 3
    assert y.val == pytest.approx([-4, -3, 2, 7])
    for d in y.der:
        assert d == pytest.approx([1, 1, 1, 1])

def test_sub_AD():
    x = AD(val=[14])
    y = x - x
    assert y.val == pytest.approx([0])
    for d in y.der:
        assert d == pytest.approx([0])

def test_sub_AD_many_val():
    x = AD(val=[-1, 0, 5, 10])
    y = x - x
    assert y.val == pytest.approx([0, 0, 0, 0])
    for d in y.der:
        assert d == pytest.approx([0, 0, 0, 0])

def test_sub_different_AD():
    x = AD(val=[3], index=0, magnitude=2)
    y = AD(val=[4], index=1, magnitude=2)
    z = x - y
    assert z.val == pytest.approx([-1])
    assert z.der[0] == pytest.approx([1])
    assert z.der[1] == pytest.approx([-1])

def test_sub_different_AD_many_val():
    x = AD(val=[10, -1, 3.2, 4], index=0, magnitude=2)
    y = AD(val=[-2, 0, 1, 100], index=1, magnitude=2)
    z = x - y
    assert z.val == pytest.approx([12, -1, 2.2, -96])
    assert z.der[0] == pytest.approx([1, 1, 1, 1])
    assert z.der[1] == pytest.approx([-1, -1, -1, -1])

def test_rsub_const():
    x = AD(val=[1])
    y = 7 - x
    assert y.val == pytest.approx([6])
    for d in y.der:
        assert d == pytest.approx([-1])

def test_rsub_const_many_val():
    x = AD(val=[-1, 0, 5, 10])
    y = 17 - x
    assert y.val == pytest.approx([18, 17, 12, 7])
    for d in y.der:
        assert d == pytest.approx([-1, -1, -1, -1])

# Testing the mul and rmul
def test_mul_const():
    x = AD(val=[10])
    y = x * 3
    assert y.val == pytest.approx([30])
    for d in y.der:
        assert d == pytest.approx([3])

def test_mul_const_many_val():
    x = AD(val=[-1, 0, 5, 10])
    y = x * 3
    assert y.val == pytest.approx([-3, 0, 15, 30])
    for d in y.der:
        assert d == pytest.approx([3, 3, 3, 3])

def test_mul_AD():
    x = AD(val=[4])
    y = x * x
    assert y.val == pytest.approx([16])
    for d in y.der:
        assert d == pytest.approx([8])

def test_mul_AD_many_val():
    x = AD(val=[-1, 0, 5, 10])
    y = x * x
    assert y.val == pytest.approx([1, 0, 25, 100])
    for d in y.der:
        assert d == pytest.approx([-2, 0, 10, 20])

def test_mul_different_AD():
    x = AD(val=[3], index=0, magnitude=2)
    y = AD(val=[4], index=1, magnitude=2)
    z = x * y
    assert z.val == pytest.approx([12])
    assert z.der[0] == pytest.approx([4])
    assert z.der[1] == pytest.approx([3])

def test_mul_different_AD_many_val():
    x = AD(val=[10, -1, 3.2, 4], index=0, magnitude=2)
    y = AD(val=[-2, 0, 1, 100], index=1, magnitude=2)
    z = x * y
    assert z.val == pytest.approx([-20, 0, 3.2, 400])
    assert z.der[0] == pytest.approx([-2, 0, 1, 100])
    assert z.der[1] == pytest.approx([10, -1, 3.2, 4])

def test_rmul_const():
    x = AD(val=[1])
    y = 7 * x
    assert y.val == pytest.approx([7])
    for d in y.der:
        assert d == pytest.approx([7])

def test_rmul_const_many_val():
    x = AD(val=[-1, 0, 5, 10])
    y = 7 * x
    assert y.val == pytest.approx([-7, 0, 35, 70])
    for d in y.der:
        assert d == pytest.approx([7, 7, 7, 7])

# Testing the div and rdiv
def test_div_const():
    x = AD(val=[20])
    y = x / 4
    assert y.val == pytest.approx([5])
    for d in y.der:
        assert d == pytest.approx([0.25])

def test_div_const_many_val():
    x = AD(val=[-1, 0, 5, 10])
    y = x / 4
    assert y.val == pytest.approx([-0.25, 0, 1.25, 2.5])
    for d in y.der:
        assert d == pytest.approx([0.25, 0.25, 0.25, 0.25])

def test_div_AD():
    x = AD(val=[4])
    y = x / x
    assert y.val == pytest.approx([1])
    for d in y.der:
        assert d == pytest.approx([0])

def test_div_AD_many_val():
    x = AD(val=[-1, 1, 5, 10])
    y = x / x
    assert y.val == pytest.approx([1, 1, 1, 1])
    for d in y.der:
        assert d == pytest.approx([0, 0, 0, 0])

def test_div_different_AD():
    x = AD(val=[2], index=0, magnitude=2)
    y = AD(val=[4], index=1, magnitude=2)
    z = x / y
    assert z.val == pytest.approx([0.5])
    assert z.der[0] == pytest.approx([0.25])
    assert z.der[1] == pytest.approx([-0.125])

def test_div_different_AD_many_val():
    x = AD(val=[-2, 4,10, 100], index=0, magnitude=2)
    y = AD(val=[1, 2, 2, 4], index=1, magnitude=2)
    z = x / y
    assert z.val == pytest.approx([-2, 2, 5, 25])
    assert z.der[0] == pytest.approx([1, 0.5, 0.5, 0.25])
    assert z.der[1] == pytest.approx([2, -1, -2.5, -6.25])

def test_rdiv_const():
    x = AD(val=[1])
    y = 7 / x
    assert y.val == pytest.approx([7])
    for d in y.der:
        assert d == pytest.approx([-7])

def test_rdiv_const_many_val():
    x = AD(val=[-1, 1, 7, 14])
    y = 7 / x
    assert y.val == pytest.approx([-7, 7, 1, 0.5])
    for d in y.der:
        assert d == pytest.approx([-7, -7, -0.14285714285714285, -0.03571428571428571])

# Testing the power and rpower
def test_pow_const():
    x = AD(val=[5])
    y = x**2
    assert y.val == pytest.approx([25])
    for d in y.der:
        assert d == pytest.approx([10])

def test_pow_const_many_val():
    x = AD(val=[-1, 0, 1, 3])
    y = x ** 3
    assert y.val == pytest.approx([-1, 0, 1, 27])
    for d in y.der:
        assert d == pytest.approx([3, 0, 3, 27])

def test_pow_AD():
    x = AD(val=[2])
    y = x ** x
    assert y.val == pytest.approx([4])
    for d in y.der:
        assert d == pytest.approx([6.772588722239782])

def test_pow_AD_many_val():
    x = AD(val=[1, 2, 5, 10])
    y = x ** x
    assert y.val == pytest.approx([1, 4, 3125, 10000000000])
    for d in y.der:
        assert d == pytest.approx([1, 6.772588722239782, 8154.493476356564, 33025850929.94046])

def test_pow_different_AD():
    x = AD(val=[2], index=0, magnitude=2)
    y = AD(val=[3], index=1, magnitude=2)
    z = x ** y
    assert z.val == pytest.approx([8])
    assert z.der[0] == pytest.approx([12])
    assert z.der[1] == pytest.approx([5.545177444479562])

def test_pow_different_AD_many_val():
    x = AD(val=[1, 2, 3, 4], index=0, magnitude=2)
    y = AD(val=[4, 3, 2, 1], index=1, magnitude=2)
    z = x ** y
    assert z.val == pytest.approx([1, 8, 9, 4])
    assert z.der[0] == pytest.approx([4, 12, 6, 1])
    assert z.der[1] == pytest.approx([0, 5.545177444479562, 9.887510598012987, 5.545177444479562])

def test_rpow_const():
    x = AD(val=[1])
    y = 2 ** x
    assert y.val == pytest.approx([2])
    for d in y.der:
        assert d == pytest.approx([1.3862943611198906])

def test_rpow_const_many_val():
    x = AD(val=[-1, 0, 1, 2])
    y = 2 ** x
    assert y.val == pytest.approx([0.5, 1, 2, 4])
    for d in y.der:
        assert d == pytest.approx([0.34657359027997264, 0.6931471805599453, 1.3862943611198906, 2.772588722239781])
