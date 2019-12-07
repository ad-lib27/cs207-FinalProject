import pytest
from adlib27.autodiff import AutoDiff as AD
import numpy as np

# Testing the getters and setters
def test_getters():
    x = AD(10,2)
    value = x.val
    derivative = x.der
    assert value == 10
    assert derivative == 2

def test_setters():
    x = AD(10,5)
    x.val = 5
    x.der = 15
    assert x.val == 5
    assert x.der == 15

# Testing the Unary operations
def test_neg():
    x = AD(10)
    y = -x
    assert y.val == -10
    assert y.der == -1

def test_pos():
    x = AD(-15)
    y = +x
    assert y.val == -15
    assert y.der == 1

# Testing the basic operations (+, -, *, /)
# Testing the add and radd
def test_add_const():
    x = AD(10)
    y = x + 22
    assert y.val == 32
    assert y.der == 1

def test_add_AD():
    x = AD(3)
    y = x + x
    assert y.val == 6
    assert y.der == 2

def test_radd_const():
    x = AD(10)
    y = 5 + x
    assert y.val == 15
    assert y.der == 1

# Testing the sub and rsub
def test_sub_const():
    x = AD(10)
    y = x - 3
    assert y.val == 7
    assert y.der == 1

def test_sub_AD():
    x1 = AD(14)
    x2 = AD(3)
    y = x1 - x2
    assert y.val == 11
    assert y.der == 0

def test_rsub_const():
    x = AD(1)
    y = 7 - x
    assert y.val == 6
    assert y.der == -1

# Testing the mul and rmul
def test_mul_const():
    x = AD(10)
    y = x*3
    assert y.val == 30
    assert y.der == 3

def test_mul_AD():
    x1 = AD(5)
    x2 = AD(3)
    y = x1 * x2
    assert y.val == 15
    assert y.der == 8

def test_rmul_const():
    x = AD(6)
    y = 2*x
    assert y.val == 12
    assert y.der == 2

# Testing the div and rdiv
def test_div_const():
    x = AD(20)
    y = x/4
    assert y.val == 5
    assert y.der == 0.25

def test_div_AD():
    x1 = AD(15)
    x2 = AD(5)
    y = x1 / x2
    assert y.val == 3
    assert y.der == -0.4

def test_rdiv_const():
    x = AD(5)
    y = 10/x
    assert y.val == 2
    assert y.der == -0.4

# Testing the power and rpower
def test_power_const():
    x = AD(5)
    y = x**2
    assert y.val == 25
    assert y.der == 10

def test_power_AD():
    x1 = AD(1)
    x2 = AD(3)
    y = x1 ** x2
    assert y.val == 1
    assert y.der == 3

def test_rpower_const():
    x = AD(2)
    y = 10**x
    assert y.val == 100
    assert y.der == 10**2* np.log(10)
