import pytest
import adlib27.elem_function as ef
from adlib27.autodiff import AutoDiff as AD
import numpy as np
import math

# one value for one variable
def test_sin1():
    # default AD object with .val=[0.0]
    x = AD()
    y = ef.sin(x)
    assert y.val == pytest.approx(np.sin(0))
    assert y.der == pytest.approx(np.cos(0) * 1)

# multiple value for one variable
def test_sin2():
    # AD object with .val=[1,2]
    x = AD(val=[1,2], index=0, magnitude=1)
    y = ef.sin(x)
    assert y.val == pytest.approx(np.sin([1,2]))
    assert y.der[0] == pytest.approx(np.cos([1,2]) * 1)

# one value for multiple variable
def test_sin3():
    # AD object with .val=[1]
    x1 = AD(val=[1], index=0, magnitude=2)
    # AD object with .val=[2]
    x2 = AD(val=[2], index=1, magnitude=2)
    y = ef.sin(x1+x2)
    assert y.val == pytest.approx(np.sin([3]))
    assert y.der[0] == pytest.approx(np.cos(3) * 1)
    assert y.der[1] == pytest.approx(np.cos(3) * 1)


# multiple value for multiple variable
def test_sin4():
    # AD object with .val=[1,2]
    x1 = AD(val=[1,2], index=0, magnitude=2)
    # AD object with .val=[2,3]
    x2 = AD(val=[2,3], index=1, magnitude=2)
    y = ef.sin(x1 + x2)
    assert y.val == pytest.approx(np.sin([3,5]))
    assert y.der[0] == pytest.approx(np.cos([3,5]) * 1)
    assert y.der[1] == pytest.approx(np.cos([3,5]) * 1)


# one value for one variable
def test_cos1():
    # default AD object with .val=[0.0]
    x = AD()
    y = ef.cos(x)
    assert y.val == pytest.approx(np.cos(0))
    assert y.der == pytest.approx(- np.sin(0) * 1)

# multiple value for one variable
def test_cos2():
    # AD object with .val=[1,2]
    x = AD(val=[1,2], index=0, magnitude=1)
    y = ef.cos(x)
    assert y.val == pytest.approx(np.cos([1,2]))
    assert y.der[0] == pytest.approx(-np.sin([1,2]) * 1)

# one value for multiple variable
def test_cos3():
    # AD object with .val=[1]
    x1 = AD(val=[1], index=0, magnitude=2)
    # AD object with .val=[2]
    x2 = AD(val=[2], index=1, magnitude=2)
    y = ef.cos(x1+x2)
    assert y.val == pytest.approx(np.cos([3]))
    assert y.der[0] == pytest.approx(-np.sin(3) * 1)
    assert y.der[1] == pytest.approx(-np.sin(3) * 1)


# multiple value for multiple variable
def test_cos4():
    # AD object with .val=[1,2]
    x1 = AD(val=[1,2], index=0, magnitude=2)
    # AD object with .val=[2,3]
    x2 = AD(val=[2,3], index=1, magnitude=2)
    y = ef.cos(x1 + x2)
    assert y.val == pytest.approx(np.cos([3,5]))
    assert y.der[0] == pytest.approx(-np.sin([3,5]) * 1)
    assert y.der[1] == pytest.approx(-np.sin([3,5]) * 1)

# one value for one variable
def test_tan1():
    # default AD object with .val=[0.0]
    x = AD()
    y = ef.tan(x)
    assert y.val == pytest.approx(np.tan(0))
    assert y.der == pytest.approx((1 / np.cos(0))**2 * 1)

# multiple value for one variable
def test_tan2():
    # AD object with .val=[1,2]
    x = AD(val=[1,2], index=0, magnitude=1)
    y = ef.tan(x)
    assert y.val == pytest.approx(np.tan([1,2]))
    assert y.der[0] == pytest.approx((1 / np.cos([1,2]))**2 * 1)

# one value for multiple variable
def test_tan3():
    # AD object with .val=[1]
    x1 = AD(val=[1], index=0, magnitude=2)
    # AD object with .val=[2]
    x2 = AD(val=[2], index=1, magnitude=2)
    y = ef.tan(x1+x2)
    assert y.val == pytest.approx(np.tan([3]))
    assert y.der[0] == pytest.approx((1 / np.cos(3))**2 * 1)
    assert y.der[1] == pytest.approx((1 / np.cos(3))**2 * 1)


# multiple value for multiple variable
def test_tan4():
    # AD object with .val=[1,2]
    x1 = AD(val=[1,2], index=0, magnitude=2)
    # AD object with .val=[2,3]
    x2 = AD(val=[2,3], index=1, magnitude=2)
    y = ef.tan(x1 + x2)
    assert y.val == pytest.approx(np.tan([3,5]))
    assert y.der[0] == pytest.approx((1 / np.cos([3,5]))**2 * 1)
    assert y.der[1] == pytest.approx((1 / np.cos([3,5]))**2 * 1)


# Inverse trig functions

# one value for one variable
def test_arcsin1():
    # default AD object with .val=[0.0]
    x = AD()
    y = ef.arcsin(x)
    assert y.val == pytest.approx(np.arcsin(0))
    assert y.der[0] == pytest.approx(1 / np.sqrt(1 - 0 **2) * 1)

# multiple value for multiple variable
def test_arcsin4():
    # AD object with .val=[1,2]
    x1 = AD(val=[0.1,0.2], index=0, magnitude=2)
    # AD object with .val=[2,3]
    x2 = AD(val=[0.2,0.3], index=1, magnitude=2)
    y = ef.arcsin(x1 + x2)
    assert y.val == pytest.approx(np.arcsin([0.3,0.5]))
    assert y.der[0] == pytest.approx([1.0482848367219182, 1.1547005383792517])
    assert y.der[1] == pytest.approx([1.0482848367219182, 1.1547005383792517])

# one value for one variable
def test_arccos1():
    # default AD object with .val=[0.0]
    x = AD(val=[0.8])
    y = ef.arccos(x)
    assert y.val == pytest.approx(np.arccos(0.8))
    assert y.der[0] == pytest.approx(-1 / np.sqrt(1 - 0.8 **2) * 1)

# multiple value for multiple variable
def test_arccos4():
    # AD object with .val=[1,2]
    x1 = AD(val=[0.3,0.4], index=0, magnitude=2)
    # AD object with .val=[2,3]
    x2 = AD(val=[0.4,0.5], index=1, magnitude=2)
    y = ef.arccos(x1 + x2)
    assert y.val == pytest.approx(np.arccos([0.7,0.9]))
    assert y.der[0] == pytest.approx([-1.4002800840280099, -2.294157338705618])
    assert y.der[1] == pytest.approx([-1.4002800840280099, -2.294157338705618])

# one value for one variable
def test_arctan1():
    # default AD object with .val=[0.0]
    x = AD(val=[0.8])
    y = ef.arctan(x)
    assert y.val == pytest.approx(np.arctan(0.8))
    assert y.der[0][0] == pytest.approx(1 / (1 + 0.8 **2) * 1)

# multiple value for multiple variable
def test_arctan4():
    # AD object with .val=[1,2]
    x1 = AD(val=[0.3,0.4], index=0, magnitude=2)
    # AD object with .val=[2,3]
    x2 = AD(val=[0.4,0.5], index=1, magnitude=2)
    y = ef.arctan(x1 + x2)
    assert y.val == pytest.approx(np.arctan([0.7,0.9]))
    assert y.der[0] == pytest.approx([0.6711409395973155, 0.5524861878453039])
    assert y.der[1] == pytest.approx([0.6711409395973155, 0.5524861878453039])

# one value for one variable
def test_exp1():
    # default AD object with .val=[0.0]
    x = AD()
    y = ef.exp(x)
    assert y.val == pytest.approx(np.exp(0))
    assert y.der == pytest.approx(np.exp(0) * 1)

# multiple value for one variable
def test_exp2():
    # AD object with .val=[1,2]
    x = AD(val=[1,2], index=0, magnitude=1)
    y = ef.exp(x)
    assert y.val == pytest.approx(np.exp([1,2]))
    assert y.der[0] == pytest.approx(np.exp([1,2]) * 1)

# one value for multiple variable
def test_exp3():
    # AD object with .val=[1]
    x1 = AD(val=[1], index=0, magnitude=2)
    # AD object with .val=[2]
    x2 = AD(val=[2], index=1, magnitude=2)
    y = ef.exp(x1+x2)
    assert y.val == pytest.approx(np.exp([3]))
    assert y.der[0] == pytest.approx(np.exp(3) * 1)
    assert y.der[1] == pytest.approx(np.exp(3) * 1)


# multiple value for multiple variable
def test_exp4():
    # AD object with .val=[1,2]
    x1 = AD(val=[1,2], index=0, magnitude=2)
    # AD object with .val=[2,3]
    x2 = AD(val=[2,3], index=1, magnitude=2)
    y = ef.exp(x1 + x2)
    assert y.val == pytest.approx(np.exp([3,5]))
    assert y.der[0] == pytest.approx(np.exp([3,5]) * 1)
    assert y.der[1] == pytest.approx(np.exp([3,5]) * 1)

# Hyperbolic functions (sinh, cosh, tanh)


# one value for one variable
def test_sinh1():
    # default AD object with .val=[0.0]
    x = AD()
    y = ef.sinh(x)
    assert y.val == pytest.approx(np.sinh(0))
    assert y.der == pytest.approx(np.cosh(0) * 1)

# multiple value for one variable
def test_sinh2():
    # AD object with .val=[1,2]
    x = AD(val=[1,2], index=0, magnitude=1)
    y = ef.sinh(x)
    assert y.val == pytest.approx(np.sinh([1,2]))
    assert y.der[0] == pytest.approx(np.cosh([1,2]) * 1)

# one value for multiple variable
def test_sinh3():
    # AD object with .val=[1]
    x1 = AD(val=[1], index=0, magnitude=2)
    # AD object with .val=[2]
    x2 = AD(val=[2], index=1, magnitude=2)
    y = ef.sinh(x1+x2)
    assert y.val == pytest.approx(np.sinh([3]))
    assert y.der[0] == pytest.approx(np.cosh(3) * 1)
    assert y.der[1] == pytest.approx(np.cosh(3) * 1)


# multiple value for multiple variable
def test_sinh4():
    # AD object with .val=[1,2]
    x1 = AD(val=[1,2], index=0, magnitude=2)
    # AD object with .val=[2,3]
    x2 = AD(val=[2,3], index=1, magnitude=2)
    y = ef.sinh(x1 + x2)
    assert y.val == pytest.approx(np.sinh([3,5]))
    assert y.der[0] == pytest.approx(np.cosh([3,5]) * 1)
    assert y.der[1] == pytest.approx(np.cosh([3,5]) * 1)

# one value for one variable
def test_cosh1():
    # default AD object with .val=[0.0]
    x = AD()
    y = ef.cosh(x)
    assert y.val == pytest.approx(np.cosh(0))
    assert y.der == pytest.approx(np.sinh(0) * 1)

# multiple value for one variable
def test_cosh2():
    # AD object with .val=[1,2]
    x = AD(val=[1,2], index=0, magnitude=1)
    y = ef.cosh(x)
    assert y.val == pytest.approx(np.cosh([1,2]))
    assert y.der[0] == pytest.approx(np.sinh([1,2]) * 1)

# one value for multiple variable
def test_cosh3():
    # AD object with .val=[1]
    x1 = AD(val=[1], index=0, magnitude=2)
    # AD object with .val=[2]
    x2 = AD(val=[2], index=1, magnitude=2)
    y = ef.cosh(x1+x2)
    assert y.val == pytest.approx(np.cosh([3]))
    assert y.der[0] == pytest.approx(np.sinh(3) * 1)
    assert y.der[1] == pytest.approx(np.sinh(3) * 1)


# multiple value for multiple variable
def test_cosh4():
    # AD object with .val=[1,2]
    x1 = AD(val=[1,2], index=0, magnitude=2)
    # AD object with .val=[2,3]
    x2 = AD(val=[2,3], index=1, magnitude=2)
    y = ef.cosh(x1 + x2)
    assert y.val == pytest.approx(np.cosh([3,5]))
    assert y.der[0] == pytest.approx(np.sinh([3,5]) * 1)
    assert y.der[1] == pytest.approx(np.sinh([3,5]) * 1)

# one value for one variable
def test_tanh1():
    # default AD object with .val=[0.0]
    x = AD()
    y = ef.tanh(x)
    assert y.val == pytest.approx(np.tanh(0))
    assert y.der == pytest.approx(1 / np.cosh(0))

# multiple value for one variable
def test_tanh2():
    # AD object with .val=[1,2]
    x = AD(val=[1,2], index=0, magnitude=1)
    y = ef.tanh(x)
    assert y.val == pytest.approx(np.tanh([1,2]))
    assert y.der[0] == pytest.approx(1 / np.cosh([1,2]))

# one value for multiple variable
def test_tanh3():
    # AD object with .val=[1]
    x1 = AD(val=[1], index=0, magnitude=2)
    # AD object with .val=[2]
    x2 = AD(val=[2], index=1, magnitude=2)
    y = ef.tanh(x1+x2)
    assert y.val == pytest.approx(np.tanh([3]))
    assert y.der[0] == pytest.approx(1 / np.cosh(3))
    assert y.der[1] == pytest.approx(1 / np.cosh(3))


# multiple value for multiple variable
def test_tanh4():
    # AD object with .val=[1,2]
    x1 = AD(val=[1,2], index=0, magnitude=2)
    # AD object with .val=[2,3]
    x2 = AD(val=[2,3], index=1, magnitude=2)
    y = ef.tanh(x1 + x2)
    assert y.val == pytest.approx(np.tanh([3,5]))
    assert y.der[0] == pytest.approx(np.sinh(1 / np.cosh([3,5])),rel=1e-2)
    assert y.der[1] == pytest.approx(np.sinh(1 / np.cosh([3,5])),rel=1e-2)


# Logistic function

# one value for one variable
def test_logistic1():
    # default AD object with .val=[0.0]
    x = AD()
    y = ef.logistic(x)
    assert y.val[0] == pytest.approx(0.5)
    assert y.der[0][0] == pytest.approx(0.25)


# multiple value for multiple variable
def test_logistic4():
    # AD object with .val=[1,2]
    x1 = AD(val=[1,2], index=0, magnitude=2)
    # AD object with .val=[2,3]
    x2 = AD(val=[2,3], index=1, magnitude=2)
    y = ef.logistic(x1 + x2)
    assert y.val == pytest.approx([0.95257412,0.99330714])
    assert y.der[0] == pytest.approx([0.045176659730, 0.006648056670])
    assert y.der[1] == pytest.approx([0.045176659730, 0.006648056670])

# one value for one variable
def test_log1():
    # default AD object with .val=[0.0]
    x = AD(val=[2])
    y = ef.log(x)
    assert y.val[0] == pytest.approx(np.log(2))
    assert y.der[0][0] == pytest.approx((1/2*1))


# multiple value for multiple variable
def test_log4():
    # AD object with .val=[1,2]
    x1 = AD(val=[1,2], index=0, magnitude=2)
    # AD object with .val=[2,3]
    x2 = AD(val=[2,3], index=1, magnitude=2)
    y = ef.log(x1 + x2)
    assert y.val == pytest.approx(np.log([3,5]))
    assert y.der[0] == pytest.approx([0.3333333333333333, 0.2])
    assert y.der[1] == pytest.approx([0.3333333333333333, 0.2])

# one value for one variable
def test_log2_1():
    # default AD object with .val=[0.0]
    x = AD(val=[2])
    y = ef.log2(x)
    assert y.val[0] == pytest.approx(np.log2(2))
    assert y.der[0][0] == pytest.approx((1/(2* np.log(2)) *1))


# multiple value for multiple variable
def test_log2_4():
    # AD object with .val=[1,2]
    x1 = AD(val=[1,2], index=0, magnitude=2)
    # AD object with .val=[2,3]
    x2 = AD(val=[2,3], index=1, magnitude=2)
    y = ef.log2(x1 + x2)
    assert y.val == pytest.approx(np.log2([3,5]))
    assert y.der[0] == pytest.approx([0.48089834696298783, 0.28853900817779266])
    assert y.der[1] == pytest.approx([0.48089834696298783, 0.28853900817779266])

# one value for one variable
def test_log10_1():
    # default AD object with .val=[0.0]
    x = AD(val=[2])
    y = ef.log10(x)
    assert y.val[0] == pytest.approx(np.log10(2))
    assert y.der[0][0] == pytest.approx((1/(2* np.log(10)) *1))


# multiple value for multiple variable
def test_log10_4():
    # AD object with .val=[1,2]
    x1 = AD(val=[1,2], index=0, magnitude=2)
    # AD object with .val=[2,3]
    x2 = AD(val=[2,3], index=1, magnitude=2)
    y = ef.log10(x1 + x2)
    assert y.val == pytest.approx(np.log10([3,5]))
    assert y.der[0] == pytest.approx([0.14476482730108392, 0.08685889638065035])
    assert y.der[1] == pytest.approx([0.14476482730108392, 0.08685889638065035])

# one value for one variable
def test_logb1():
    # default AD object with .val=[0.0]
    x = AD(val=[2])
    y = ef.logb(x,3)
    assert y.val[0] == pytest.approx(math.log(2,3))
    assert y.der[0][0] == pytest.approx((1/(2* np.log(3)) *1))


# multiple value for multiple variable
def test_logb4():
    # AD object with .val=[1,2]
    x1 = AD(val=[1,2], index=0, magnitude=2)
    # AD object with .val=[2,3]
    x2 = AD(val=[2,3], index=1, magnitude=2)
    y = ef.logb(x1 + x2,3)
    assert y.val[0] == pytest.approx(math.log(3,3))
    assert y.val[1] == pytest.approx(math.log(5, 3))
    assert y.der[0] == pytest.approx([0.30341307554227914, 0.18204784532536747])
    assert y.der[1] == pytest.approx([0.30341307554227914, 0.18204784532536747])

# one value for one variable
def test_sqrt1():
    # default AD object with .val=[0.0]
    x = AD(val=[2])
    y = ef.sqrt(x)
    assert y.val == pytest.approx(np.sqrt(2))
    assert y.der[0] == pytest.approx(0.5/np.sqrt(2) *1)


# multiple value for multiple variable
def test_sqrt4():
    # AD object with .val=[1,2]
    x1 = AD(val=[1,2], index=0, magnitude=2)
    # AD object with .val=[2,3]
    x2 = AD(val=[2,3], index=1, magnitude=2)
    y = ef.sqrt(x1 + x2)
    assert y.val == pytest.approx(np.sqrt([3,5]))
    assert y.der[0] == pytest.approx(0.5/np.sqrt([3,5]) *1)
    assert y.der[1] == pytest.approx(0.5/np.sqrt([3,5]) *1)
