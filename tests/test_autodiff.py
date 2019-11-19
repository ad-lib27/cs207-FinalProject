import pytest
from adlib.autodiff import AutoDiffToy as ADT

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
def test_add_const():
    x = ADT(10)
    y = x + 22
    assert y.val == 32
    assert y.der == 1

def test_add_adt():
    x = ADT(3)
    y = x + x
    assert y.val == 6
    assert y.der == 1

def test_radd_const():
    x = ADT(10)
    y = 5 + x
    assert y.val == 15
    assert y.der == 1

