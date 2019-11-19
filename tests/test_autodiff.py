import pytest
from adlib.autodiff import AutoDiffToy as ADT

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
