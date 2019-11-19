import pytest
from adlib.autodiff import AutoDiffToy as ADT
import numpy as np

# Testing the getters and setters
def test_getters():
    x = ADT(10,2)
    value = x.get_val()
    derivative = x.get_der()
    assert value == 10
    assert derivative == 2

def test_setters():
    x = ADT(10,5)
    x.set_val(5)
    x.set_der(15) 
    assert x.val == 5
    assert x.der == 15

# Testing the Unary operations
def test_neg():
    x = ADT(10)
    y = -x
    assert y.val == -10
    assert y.der == -1

def test_pos():
    x = ADT(-15)
    y = +x
    assert y.val == -15
    assert y.der == 1

# Testing the basic operations (+, -, *, /)
# Testing the add and radd
def test_add_const():
    x = ADT(10)
    y = x + 22
    assert y.val == 32
    assert y.der == 1

def test_add_adt():
    x = ADT(3)
    y = x + x
    assert y.val == 6
    assert y.der == 2

def test_radd_const():
    x = ADT(10)
    y = 5 + x
    assert y.val == 15
    assert y.der == 1

# Testing the sub and rsub
def test_sub_const():
    x = ADT(10)
    y = x - 3
    assert y.val == 7
    assert y.der == 1

def test_sub_adt():
    x1 = ADT(14)
    x2 = ADT(3)
    y = x1 - x2
    assert y.val == 11
    assert y.der == 0

def test_rsub_const():
    x = ADT(1)
    y = 7 - x
    assert y.val == 6
    assert y.der == -1

# Testing the mul and rmul
def test_mul_const():
    x = ADT(10)
    y = x*3
    assert y.val == 30
    assert y.der == 3

def test_mul_adt():
    x1 = ADT(5)
    x2 = ADT(3)
    y = x1 * x2
    assert y.val == 15
    assert y.der == 8 

def test_rmul_const():
    x = ADT(6)
    y = 2*x
    assert y.val == 12
    assert y.der == 2

# Testing the div and rdiv
def test_div_const():
    x = ADT(20)
    y = x/4
    assert y.val == 5
    assert y.der == 0.25

def test_div_adt():
    x1 = ADT(15)
    x2 = ADT(5)
    y = x1 / x2
    assert y.val == 3
    assert y.der == -0.4 

def test_rdiv_const():
    x = ADT(5)
    y = 10/x
    assert y.val == 2
    assert y.der == -0.4

# Testing the power and rpower
def test_power_const():
    x = ADT(5)
    y = x**2
    assert y.val == 25
    assert y.der == 10

def test_power_adt():
    x1 = ADT(1)
    x2 = ADT(3)
    y = x1 ** x2
    assert y.val == 1
    assert y.der == 3 

def test_rpower_const():
    x = ADT(2)
    y = 10**x
    assert y.val == 100
    assert y.der == 10**2* np.log(10)
