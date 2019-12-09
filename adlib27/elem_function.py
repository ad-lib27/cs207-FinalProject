import numpy as np

from adlib27.autodiff import AutoDiff as AD

def sin(x):
    """
    :param x: a scalar or an AD object
    :return: value and derivative of sin(x)
    """
    if isinstance(x, AD):
        der = list()
        for i in range(len(x.der)):
            der += [list()]
            for j in range(len(x.der[i])):
                der[i] += [np.cos(x.val[j]) * x.der[i][j]]
        val = list()
        for i in range(len(x.val)):
            val += [np.sin(x.val[i])]

        y = AD(val=val, der=der)
        return y
    else:
        return np.sin(x)

def cos(x):
    """
    :param x: a scalar or a function
    :return: value and derivative of cos(x)
    """
    if isinstance(x, AD):
        der = list()
        for i in range(len(x.der)):
            der += [list()]
            for j in range(len(x.der[i])):
                der[i] += [- np.sin(x.val[j]) * x.der[i][j]]
        val = list()
        for i in range(len(x.val)):
            val += [np.cos(x.val[i])]

        y = AD(val=val, der=der)
        return y
    else:
        return np.cos(x)

def tan(x):
    """
    :param x: a scalar or a function
    :return: value and derivative of tan(x)
    """
    if isinstance(x, AD):
        der = list()
        for i in range(len(x.der)):
            der += [list()]
            for j in range(len(x.der[i])):
                der[i] += [(1 / np.cos(x.val[j]))**2 * x.der[i][j]]
        val = list()
        for i in range(len(x.val)):
            val += [np.tan(x.val[i])]

        y = AD(val=val, der=der)
        return y
    else:
        return np.tan(x)

# Inverse trig functions
def arcsin(x):
    """
    :param x: a scalar or an AD object
    :return: value and derivative of arcsin(x)
    """
    if isinstance(x, AD):
        der = list()
        for i in range(len(x.der)):
            der += [list()]
            for j in range(len(x.der[i])):
                der[i] += [1 / np.sqrt(1 - x.val[j] **2) * x.der[i][j]]
        val = list()
        for i in range(len(x.val)):
            val += [np.arcsin(x.val[i])]

        y = AD(val=val, der=der)
        return y
    else:
        return np.arcsin(x)


def arccos(x):
    """
    :param x: a scalar or an AD object
    :return: value and derivative of arccos(x)
    """
    if isinstance(x, AD):
        der = list()
        for i in range(len(x.der)):
            der += [list()]
            for j in range(len(x.der[i])):
                der[i] += [-1 / np.sqrt(1 - x.val[j] **2) * x.der[i][j]]
        val = list()
        for i in range(len(x.val)):
            val += [np.arccos(x.val[i])]

        y = AD(val=val, der=der)
        return y
    else:
        return np.arccos(x)


def arctan(x):
    """
    :param x: a scalar or an AD object
    :return: value and derivative of arctan(x)
    """
    if isinstance(x, AD):
        der = list()
        for i in range(len(x.der)):
            der += [list()]
            for j in range(len(x.der[i])):
                der[i] += [1 / (1 + x.val[j] **2) * x.der[i][j]]
        val = list()
        for i in range(len(x.val)):
            val += [np.arctan(x.val[i])]

        y = AD(val=val, der=der)
        return y
    else:
        return np.arctan(x)


# Exponentials
def exp(x):
    """
    :param x: a scalar or an AD object
    :return: value and derivative of exp(x)
    """
    if isinstance(x, AD):
        der = list()
        for i in range(len(x.der)):
            der += [list()]
            for j in range(len(x.der[i])):
                der[i] += [np.exp(x.val[j]) * x.der[i][j]]
        val = list()
        for i in range(len(x.val)):
            val += [np.exp(x.val[i])]

        y = AD(val=val, der=der)
        return y
    else:
        return np.exp(x)

# Hyperbolic functions (sinh, cosh, tanh)
def sinh(x):
    """
    :param x: a scalar or an AD object
    :return: value and derivative of sinh(x)
    """
    if isinstance(x, AD):
        der = list()
        for i in range(len(x.der)):
            der += [list()]
            for j in range(len(x.der[i])):
                der[i] += [np.cosh(x.val[j]) * x.der[i][j]]
        val = list()
        for i in range(len(x.val)):
            val += [np.sinh(x.val[i])]

        y = AD(val=val, der=der)
        return y
    else:
        return np.sinh(x)

def cosh(x):
    """
    :param x: a scalar or an AD object
    :return: value and derivative of cosh(x)
    """
    if isinstance(x, AD):
        der = list()
        for i in range(len(x.der)):
            der += [list()]
            for j in range(len(x.der[i])):
                der[i] += [np.sinh(x.val[j]) * x.der[i][j]]
        val = list()
        for i in range(len(x.val)):
            val += [np.cosh(x.val[i])]

        y = AD(val=val, der=der)
        return y
    else:
        return np.cosh(x)


def tanh(x):
    """
    :param x: a scalar or an AD object
    :return: value and derivative of tanh(x)
    """
    if isinstance(x, AD):
        der = list()
        for i in range(len(x.der)):
            der += [list()]
            for j in range(len(x.der[i])):
                der[i] += [x.der[i][j] / np.cosh(x.val[j])]
        val = list()
        for i in range(len(x.val)):
            val += [np.tanh(x.val[i])]

        y = AD(val=val, der=der)
        return y
    else:
        return np.tanh(x)

# Logistic function

def logistic(x):
    """
    :param x: a scalar or an AD object
    :return: value and derivative of logistic(x)
    """
    if isinstance(x, AD):
        der = list()
        for i in range(len(x.der)):
            der += [list()]
            for j in range(len(x.der[i])):
                der[i] += [(1/(1+np.exp(-x.val[j])))*(1-1/(1+np.exp(-x.val[j])))*x.der[i][j]]
        val = list()
        for i in range(len(x.val)):
            val += [1/(1+np.exp(-x.val[i]))]

        y = AD(val=val, der=der)
        return y
    else:
        return 1/(1+np.exp(-x))

# Logarithms
def log(x):
    """
    :param x: a scalar or an AD object
    :return: value and derivative of log(x)
    """
    if isinstance(x, AD):
        der = list()
        for i in range(len(x.der)):
            der += [list()]
            for j in range(len(x.der[i])):
                der[i] += [(1/x.val[j])*x.der[i][j]]
        val = list()
        for i in range(len(x.val)):
            val += [np.log(x.val[i])]

        y = AD(val=val, der=der)
        return y
    else:
        return np.log(x)

def log2(x):
    """
    :param x: a scalar or an AD object
    :return: value and derivative of log2(x)
    """
    if isinstance(x, AD):
        der = list()
        for i in range(len(x.der)):
            der += [list()]
            for j in range(len(x.der[i])):
                der[i] += [(1/(x.val[j]*np.log(2)))*x.der[i][j]]
        val = list()
        for i in range(len(x.val)):
            val += [np.log2(x.val[i])]

        y = AD(val=val, der=der)
        return y
    else:
        return np.log2(x)


def log10(x):
    """
    :param x: a scalar or an AD object
    :return: value and derivative of log10(x)
    """
    if isinstance(x, AD):
        der = list()
        for i in range(len(x.der)):
            der += [list()]
            for j in range(len(x.der[i])):
                der[i] += [(1/(x.val[j]*np.log(10)))*x.der[i][j]]
        val = list()
        for i in range(len(x.val)):
            val += [np.log10(x.val[i])]

        y = AD(val=val, der=der)
        return y
    else:
        return np.log10(x)

def logb(x, base):
    """
    :param x: a scalar or an AD object
    :return: value and derivative of logb(x) with base
    """
    if isinstance(x, AD):
        der = list()
        for i in range(len(x.der)):
            der += [list()]
            for j in range(len(x.der[i])):
                der[i] += [(1/(x.val[j]*np.log(base)))*x.der[i][j]]
        val = list()
        for i in range(len(x.val)):
            val += [np.log(x.val[i]) / np.log(base)]

        y = AD(val=val, der=der)
        return y
    else:
        return np.log(x) / np.log(base)
#Square root

def sqrt(x):
    """
    :param x: a scalar or an AD object
    :return: value and derivative of sqrt(x) with base
    """
    if isinstance(x, AD):
        der = list()
        for i in range(len(x.der)):
            der += [list()]
            for j in range(len(x.der[i])):
                der[i] += [0.5/np.sqrt(x.val[j]) * x.der[i][j]]
        val = list()
        for i in range(len(x.val)):
            val += [np.sqrt(x.val[i])]

        y = AD(val=val, der=der)
        return y
    else:
        return np.sqrt(x)
