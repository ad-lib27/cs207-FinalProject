import pytest
from code.autodiff import AutoDiffToy as ADT


# Testing the Unary operations
def test_neg():
    x = ADT(10)
    y = -x
    assert y.val == -10
    assert y.der == -1

def test_pos():
    x = ADT(-15)
    f = +x
    assert f.val == -15
    assert f.der == 1
