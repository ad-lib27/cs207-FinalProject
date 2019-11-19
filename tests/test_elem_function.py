import pytest
import elem_function as ef
from autodiff import AutoDiffToy as ADT
import numpy as np

def test_exp1():
    #default ADT object with .val=0 and .der=1
    x = ADT()
    y = ef.exp(x)
    assert y.val == np.exp(0)
    assert y.der == np.exp(0) * 1

def test_exp2():
    #default ADT object with .val=0 and .der=1
    x = 1
    assert ef.exp(x) == np.exp(x)

def test_sin1():
    x = ADT()
    y = ef.sin(x)
    assert y.val == np.sin(0)
    assert y.der == np.cos(0) * 1

def test_sin2():
    x = 1
    assert ef.sin(x) == np.sin(x)

def test_cos1():
    x = ADT()
    y = ef.cos(x)
    assert y.val == np.cos(0)
    assert y.der == - np.sin(0) * 1

def test_cos2():
    x = 1
    assert ef.cos(x) == np.cos(x)

def test_tan1():
    x = ADT()
    y = ef.tan(x)
    assert y.val == np.tan(0)
    assert y.der == (1 / np.cos(0))**2 * 1

def test_tan2():
    x = 1
    assert ef.tan(x) == np.tan(x)