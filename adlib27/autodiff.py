import numpy as np
import pytest

class AutoDiff:
    """
    Attributes:
        self.val [float]: The value of the AutoDiff number
        self.der [float]: The value of the derivative of the AutoDiff number
    """
    def __init__(self, val=[0.0], index=0, magnitude=1, der=None):
        self.val = val
        if der is None:
            self.vector_index = index
            self.vector_magnitude = magnitude
            self.der = list()
            for i in range(self.vector_magnitude):
                self.der += [[0.0 for j in range(len(self.val))]]
            self.der[self.vector_index] = [1.0 for j in range(len(self.val))]
        else:
            self.der = der


    # Comparison operations
    def __eq__(self, other):
        """
        Overloading the equality operator
        Parameters:
            self (AutoDiff): The AutoDiff number itself
            other (AutoDiff): The other AutoDiff number to compare
        Returns:
            bool: True if both AutoDiffs are equivalent, False otherwise
        """
        if isinstance(other, AutoDiff):
            try:
                assert self.val == pytest.approx(other.val)
                for s, o in zip(self.der, other.der):
                    assert s == pytest.approx(o)
                return True
            except AssertionError:
                return False
        return False


    def __ne__(self, other):
        """
        Overloading the inequality operator
        Parameters:
            self (AutoDiff): The AutoDiff number itself
            other (AutoDiff): The other AutoDiff number to compare
        Returns:
            bool: True if both AutoDiffs are different, False if they're equivalent
        """
        return not self.__eq__(other)

    # Unary operations (negation)

    def __neg__(self):
        """
        Overloading the negation operator
        Parameters:
            self (AutoDiff): The AutoDiff number itself
        Returns:
            AutoDiff: A new AutoDiff number after the negation
        """
        # val and der of the negation
        val = list()
        der = list()

        # negate the vals
        for i in range(len(self.val)):
            val += [-self.val[i]]

        # negate the ders
        for i in range(len(self.der)):
            der += [list()]
            for j in range(len(self.der[i])):
                der[i] += [-self.der[i][j]]

        return AutoDiff(val=val, der=der)

    def __pos__(self):
        """
        Overloading the unary + operator
        Parameters:
            self (AutoDiff): The AutoDiff number itself
        Returns:
            AutoDiff: The same, unchanged AutoDiff number
        """
        return self

    # Basic arithmetic operations (+, -, *, /)

    def __add__(self, other):
        """
        Overloading the addition operator
        parameters:
            self (AutoDiff): The AutoDiff number itself
            other (AutoDiff/float): The other number to add
        returns:
            AutoDiff: A new AutoDiff number with the sum of the numbers
        """
        # val and der of the sum
        val = list()
        der = list()

        # if both numbers are AutoDiff...
        if isinstance(other, AutoDiff):
            try:
                # check if `other` is a valid AutoDiff number
                assert(len(self.val) == len(other.val))

                # add their vals
                for i in range(len(self.val)):
                    val += [self.val[i] + other.val[i]]

                # add their ders
                for i in range(len(self.der)):
                    der += [list()]
                    for j in range(len(self.der[i])):
                        der[i] += [self.der[i][j] + other.der[i][j]]
            except:
                raise ValueError("Can't add AutoDiff numbers of different dimensions.")
        else:
            # add the vals of self with `other`
            for i in range(len(self.val)):
                val += [self.val[i] + other]

            # ders remain unchanged
            der = self.der

        # return sum
        return AutoDiff(val=val, der=der)

    def __radd__(self, other):
        """
        Overloading the addition in case the addition of a object/non-object will be from the left
        parameters:
            self (AutoDiff): The AutoDiff number itself
            other (AutoDiff/float): The other number to add
        returns:
            AutoDiff: A new AutoDiff number with the sum of the numbers
        """
        return self + other

    def __sub__(self, other):
        """
        Overloading the subtraction operator
        parameters:
            self (AutoDiff): The AutoDiff number itself
            other (AutoDiff/float): The other number to subtract from self
        returns:
            AutoDiff: A new AutoDiff number with the delta between the numbers
        """
        # val and der of the difference
        val = list()
        der = list()

        # if both numbers are AutoDiff...
        if isinstance(other, AutoDiff):
            try:
                # check if `other` is a valid AutoDiff number
                assert(len(self.val) == len(other.val))

                # subtract their vals
                for i in range(len(self.val)):
                    val += [self.val[i] - other.val[i]]

                # subtract their ders
                for i in range(len(self.der)):
                    der += [list()]
                    for j in range(len(self.der[i])):
                        der[i] += [self.der[i][j] - other.der[i][j]]
            except:
                raise ValueError("Can't subtract AutoDiff numbers of different dimensions.")
        else:
            # subtract the vals of self with `other`
            for i in range(len(self.val)):
                val += [self.val[i] - other]

            # ders remain unchanged
            der = self.der

        # return difference
        return AutoDiff(val=val, der=der)

    def __rsub__(self, other):
        """
        Overloading the subtraction operator in case the subtraction of a object/non-object will be from the left
        parameters:
            self (AutoDiff): The AutoDiff number itself
            other (AutoDiff/float): The other number from which to subtract
        returns:
            AutoDiff: A new AutoDiff number with the delta between the numbers
        """
        # __rsub__ will only be called if the lefthand value of the subtraction is not an AutoDiff

        # val and der of the difference
        val = list()
        der = list()

        # subtract the vals of self from `other`
        for i in range(len(self.val)):
            val += [other - self.val[i]]

        # ders are negated
        for i in range(len(self.der)):
            der += [list()]
            for j in range(len(self.der[i])):
                der[i] += [-self.der[i][j]]

        # return difference
        return AutoDiff(val=val, der=der)

    def __mul__(self, other):
        """
        Overloading the multiplication operator
        parameters:
            self (AutoDiff): The AutoDiff number itself
            other (AutoDiff/float): The other number to multiply
        returns:
            AutoDiff: A new AutoDiff number with the multiplication of the numbers
        """
        # val and der of the product
        val = list()
        der = list()

        # if both numbers are AutoDiff...
        if isinstance(other, AutoDiff):
            try:
                # check if `other` is a valid AutoDiff number
                assert(len(self.val) == len(other.val))

                # multiply their vals
                for i in range(len(self.val)):
                    val += [self.val[i] * other.val[i]]

                # update ders with product rule
                for i in range(len(self.der)):
                    der += [list()]
                    for j in range(len(self.der[i])):
                        der[i] += [(self.val[j] * other.der[i][j]) + (self.der[i][j] * other.val[j])]
            except:
                raise ValueError("Can't multiply AutoDiff numbers of different dimensions.")
        else:
            # multiply the vals of self with `other`
            for i in range(len(self.val)):
                val += [self.val[i] * other]

            # multiply the ders with `other`
            for i in range(len(self.der)):
                der += [list()]
                for j in range(len(self.der[i])):
                    der[i] += [self.der[i][j] * other]

        # return product
        return AutoDiff(val=val, der=der)

    # Implementing __rmul__ in case a non AutoDiff object will be on the left of a multiplication
    def __rmul__(self, other):
        """
        Overloading the multiplication operator in case the mul of a object/non-object will be from the left
        parameters:
            self (AutoDiff): The AutoDiff number itself
            other (AutoDiff/float): The other number to multiply
        returns:
            AutoDiff: A new AutoDiff number with the multiplication of the numbers
        """
        return self * other

    def __truediv__(self, other):
        """
        Overloading the division operator
        parameters:
            self (AutoDiff): The AutoDiff number itself
            other (AutoDiff/float): The other number by which to divide
        returns:
            AutoDiff: A new AutoDiff number with the quotient of the numbers
        """
        # val and der of the quotient
        val = list()
        der = list()

        # if both numbers are AutoDiff...
        if isinstance(other, AutoDiff):
            try:
                # check if `other` is a valid AutoDiff number
                assert(len(self.val) == len(other.val))

                # divide their vals
                for i in range(len(self.val)):
                    val += [self.val[i] / other.val[i]]

                # update ders with quotient rule
                for i in range(len(self.der)):
                    der += [list()]
                    for j in range(len(self.der[i])):
                        der[i] += [((other.val[j] * self.der[i][j]) - (self.val[j] * other.der[i][j])) / (other.val[j] ** 2)]
            except:
                raise ValueError("Can't divide AutoDiff numbers of different dimensions.")
        else:
            # divide the vals of self with `other`
            for i in range(len(self.val)):
                val += [self.val[i] / other]

            # divide the ders with `other`
            for i in range(len(self.der)):
                der += [list()]
                for j in range(len(self.der[i])):
                    der[i] += [self.der[i][j] / other]

        # return quotient
        return AutoDiff(val=val, der=der)

    def __rtruediv__(self, other):
        """
        Overloading the division operator in case the truediv of a object/non-object will be from the left
        parameters:
            self (AutoDiff): The AutoDiff number itself by which to divide
            other (AutoDiff/float): The other number
        returns:
            AutoDiff: A new AutoDiff number with the quotient of the numbers
        """
        # __rtruediv__ will only be called if the lefthand value of the division is not an AutoDiff

        # val and der of the quotient
        val = list()
        der = list()

        # divide the vals of self from `other`
        for i in range(len(self.val)):
            val += [other / self.val[i]]

        # update the ders
        for i in range(len(self.der)):
            der += [list()]
            for j in range(len(self.der[i])):
                der[i] += [(-other / self.val[j] ** 2) * self.der[i][j]]

        # return quotient
        return AutoDiff(val=val, der=der)

    def __pow__(self, other):
        """
        Overloading the pow operator
        parameters:
            self (AutoDiff): The AutoDiff number itself
            other (AutoDiff or float): The other number which serves as the exponent
        returns:
            AutoDiff: A new AutoDiff number with the result of raising self to the power of other
        """
        # val and der of the exponential
        val = list()
        der = list()

        # if both numbers are AutoDiff...
        if isinstance(other, AutoDiff):
            try:
                # check if `other` is a valid AutoDiff number
                assert(len(self.val) == len(other.val))

                # for all vals, raise self to power of `other`
                for i in range(len(self.val)):
                    val += [self.val[i] ** other.val[i]]

                # update ders
                for i in range(len(self.der)):
                    der += [list()]
                    for j in range(len(self.der[i])):
                        der[i] += [(self.val[j] ** (other.val[j] - 1)) * ((other.val[j] * self.der[i][j]) + (self.val[j] * np.log(self.val[j]) * other.der[i][j]))]
            except:
                raise ValueError("Can't calculate exponential with AutoDiff numbers of different dimensions.")
        else:
            # raise the vals of self to power of `other`
            for i in range(len(self.val)):
                val += [self.val[i] ** other]

            # update the ders
            for i in range(len(self.der)):
                der += [list()]
                for j in range(len(self.der[i])):
                    der[i] += [other * (self.val[j] ** (other - 1)) * self.der[i][j]]

        # return exponential
        return AutoDiff(val=val, der=der)

    def __rpow__(self, other):
        """
        Overloading the pow operator in case the pow of a object/non-object will be from the left
        parameters:
            self (AutoDiff): The AutoDiff number itself, which serves as the exponent
            other (AutoDiff/float): The other number which serves as the base
        returns:
            AutoDiff: A new AutoDiff number with the result of raising self to the power of other
        """
        # __rpow__ will only be called if the base is not an AutoDiff but the exponent is

        # val and der of the exponential
        val = list()
        der = list()

        # raise the vals of self to the power of `other`
        for i in range(len(self.val)):
            val += [other ** self.val[i]]

        # update the ders
        for i in range(len(self.der)):
            der += [list()]
            for j in range(len(self.der[i])):
                der[i] += [np.log(other) * (other ** self.val[j]) * self.der[i][j]]

        # return exponential
        return AutoDiff(val=val, der=der)
