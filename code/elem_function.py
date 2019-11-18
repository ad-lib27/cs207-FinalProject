import numpy as np
from autodiff import AutoDiffToy as ADT

def exp(x):
    """
    :param x: a scalar or a function
    :return: value and derivative of exp(x)
    """
    if isinstance(x, ADT):
        y = ADT()
        y.der = np.exp(y.val) * y.der
        y.val = np.exp(y.val)
        return y
    else:
        return np.exp(x)


def sin(x):
    """
    :param x: a scalar or a function
    :return: value and derivative of sin(x)
    """
    if isinstance(x, ADT):
        y = ADT()
        y.der = np.cos(y.val) * y.der
        y.val = np.sin(y.val)
        return y
    else:
        return np.sin(x)

def cos(x):
    """
    :param x: a scalar or a function
    :return: value and derivative of cos(x)
    """
    if isinstance(x, ADT):
        y = ADT()
        y.der = - np.sin(y.val) * y.der
        y.val = np.cos(y.val)
        return y
    else:
        return np.cos(x)

def tan(x):
    """
    :param x: a scalar or a function
    :return: value and derivative of tan(x)
    """
    if isinstance(x, ADT):
        y = ADT()
        y.der = (1 / np.cos(y.val))**2 * y.der
        y.val = np.tan(y.val)
        return y
    else:
        return np.tan(x)
