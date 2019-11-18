import numpy as np
from autodiff import AutoDiffToy as ADT

def exp(x):
    """
    :param x: a scalar or a function
    :return: value and derivative of exp(x)
    """
    if isinstance(x, ADT):
        x.der = np.exp(x.val) * x.der
        x.val = np.exp(x.val)
        return x
    else:
        return np.exp(x)


def sin(x):
    """
    :param x: a scalar or a function
    :return: value and derivative of sin(x)
    """
    if isinstance(x, ADT):
        x.der = np.cos(x.val) * x.der
        x.val = np.sin(x.val)
        return x
    else:
        return np.sin(x)

def cos(x):
    """
    :param x: a scalar or a function
    :return: value and derivative of cos(x)
    """
    if isinstance(x, ADT):
        x.der = - np.sin(x.val) * x.der
        x.val = np.cos(x.val)
        return x
    else:
        return np.cos(x)

def tan(x):
    """
    :param x: a scalar or a function
    :return: value and derivative of tan(x)
    """
    if isinstance(x, ADT):
        x.der = (1 / np.cos(x.val))**2 * x.der
        x.val = np.tan(x.val)
        return x
    else:
        return np.tan(x)