import numpy as np
from adlib.autodiff import AutoDiffToy as ADT

def exp(x):
    """
    :param x: a scalar or a function
    :return: value and derivative of exp(x)
    """
    if isinstance(x, ADT):
        y = ADT()
        y.der = np.exp(x.val) * x.der
        y.val = np.exp(x.val)
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
        y.der = np.cos(x.val) * x.der
        y.val = np.sin(x.val)
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
        y.der = - np.sin(x.val) * x.der
        y.val = np.cos(x.val)
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
        y.der = (1 / np.cos(x.val))**2 * x.der
        y.val = np.tan(x.val)
        return y
    else:
        return np.tan(x)
