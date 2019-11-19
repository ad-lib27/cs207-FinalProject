import numpy as np

class AutoDiffToy:
    """
    Attributes:
        self.val [float]: The value of the AutoDiff number
        self.der [float]: The value of the derivative of the AutoDiff number
    """
    def __init__(self, val=0.0, der=1.0):
        self.val = val
        self.der = der

    # Getters and Setters

    def get_val(self):
        return self.val

    def set_val(self, val):
        self.val = val

    def get_der(self):
        return self.der

    def set_der(self, val):
        self.der = val

    #  Unary operations (negation)

    def __neg__(self):
        """
        Overloading the negation operator
        Parameters:
            self (AutoDiffToy): The AutoDiffToy number itself
        Returns:
            AutoDiffToy: A new AutoDiffToy number after the negation
        """
        return AutoDiffToy(-self.val, -self.der)

    def __pos__(self):
        """
        Overloading the unary + operator
        Parameters:
            self (AutoDiffToy): The AutoDiffToy number itself
        Returns:
            AutoDiffToy: The same, unchanged AutoDiffToy number
        """
        return self

    # Basic operations (+, -, *, /)

    def __add__(self, other):
        """
        Overloading the addition operator
        parameters:
            self (AutoDiffToy): The AutoDiffToy number itself
            other (AutoDiffToy/float): The other number to add
        returns:
            AutoDiffToy: A new AutoDiffToy number with the sum of the numbers
        """
        try: # This should work if both are of AutoDiffToy class
            return AutoDiffToy(self.val + other.val, self.der + other.der)
        except AttributeError: # If not, we will catch the error and add the following way
            return AutoDiffToy(self.val + other, self.der)

    def __radd__(self, other):
        """
        Overloading the addition in case the addition of a object/non-object will be from the left
        parameters:
            self (AutoDiffToy): The AutoDiffToy number itself
            other (AutoDiffToy/float): The other number to add
        returns:
            AutoDiffToy: A new AutoDiffToy number with the sum of the numbers
        """
        return self + other

    def __sub__(self, other):
        """
        Overloading the subtraction operator
        parameters:
            self (AutoDiffToy): The AutoDiffToy number itself
            other (AutoDiffToy/float): The other number to subtract from self
        returns:
            AutoDiffToy: A new AutoDiffToy number with the delta between the numbers
        """
        try: # This should work if both are of AutoDiffToy class
            return AutoDiffToy(self.val - other.val, self.der - other.der)
        except AttributeError: # If not, we will catch the error and subtract the following way
            return AutoDiffToy(self.val - other, self.der)

    def __rsub__(self, other):
        """
        Overloading the subtraction operator in case the subtraction of a object/non-object will be from the left
        parameters:
            self (AutoDiffToy): The AutoDiffToy number itself
            other (AutoDiffToy/float): The other number from which to subtract
        returns:
            AutoDiffToy: A new AutoDiffToy number with the delta between the numbers
        """
        # __rsub__ will only be called if the lefthand value of the subtraction is not an AutoDiffToy
        return AutoDiffToy(other - self.val, -self.der)

    def __mul__(self, other):
        """
        Overloading the multiplication operator
        parameters:
            self (AutoDiffToy): The AutoDiffToy number itself
            other (AutoDiffToy/float): The other number to multiply
        returns:
            AutoDiffToy: A new AutoDiffToy number with the multiplication of the numbers
        """
        try: # This should work if both are of AutoDiffToy class H(x) = F(x)* G(X)
            return AutoDiffToy(self.val * other.val, self.val * other.der + self.der * other.val)
        except AttributeError: # If not, we will catch the error and multiply the following way
            return AutoDiffToy(self.val * other, self.der * other)

    # Implementing __rmul__ in case a non AutoDiffToy object will be on the left of a multiplication
    def __rmul__(self, other):
        """
        Overloading the multiplication operator in case the mul of a object/non-object will be from the left
        parameters:
            self (AutoDiffToy): The AutoDiffToy number itself
            other (AutoDiffToy/float): The other number to multiply
        returns:
            AutoDiffToy: A new AutoDiffToy number with the multiplication of the numbers
        """
        return self * other

    def __truediv__(self, other):
        """
        Overloading the division operator
        parameters:
            self (AutoDiffToy): The AutoDiffToy number itself
            other (AutoDiffToy/float): The other number by which to divide
        returns:
            AutoDiffToy: A new AutoDiffToy number with the quotient of the numbers
        """
        try: # This should work if both are of AutoDiffToy class H(x) = F(x)/ G(X)
            new_der = (other.val * self.der - self.val * other.der) / (other.val ** 2)
            return AutoDiffToy(self.val / other.val, new_der)
        except AttributeError: # If not, we will catch the error and multiply the following way
            return AutoDiffToy(self.val / other, self.der / other)

    def __rtruediv__(self, other):
        """
        Overloading the division operator in case the truediv of a object/non-object will be from the left
        parameters:
            self (AutoDiffToy): The AutoDiffToy number itself by which to divide
            other (AutoDiffToy/float): The other number
        returns:
            AutoDiffToy: A new AutoDiffToy number with the quotient of the numbers
        """
        # __rtruediv__ will only be called if the lefthand value of the division is not an AutoDiffToy
        return AutoDiffToy(other / self.val, (-other / self.val ** 2) * self.der)

    def __pow__(self, other):
        """
        Overloading the pow operator
        parameters:
            self (AutoDiffToy): The AutoDiffToy number itself
            other (AutoDiffToy or float): The other number which serves as the exponent
        returns:
            AutoDiffToy: A new AutoDiffToy number with the result of raising self to the power of other
        """
        try: # This should work if both are of AutoDiffToy class H(x) = F(x) ** G(X)
            left = self.val ** (other.val - 1)
            right = other.val * self.der + self.val * np.log(self.val) * other.der
            new_der = left * right
            return AutoDiffToy(self.val ** other.val, new_der)
        except AttributeError: # If not, we will catch the error and do the exponent the following way
            return AutoDiffToy(self.val ** other, other * self.val * self.der)

    def __rpow__(self, other):
        """
        Overloading the pow operator in case the pow of a object/non-object will be from the left
        parameters:
            self (AutoDiffToy): The AutoDiffToy number itself, which serves as the exponent
            other (AutoDiffToy/float): The other number which serves as the base
        returns:
            AutoDiffToy: A new AutoDiffToy number with the result of raising self to the power of other
        """
        # __rpow__ will only be called if the base is not an AutoDiffToy but the exponent is
        return AutoDiffToy(other ** self.val, np.log(other) * (other ** self.val) * self.der)
