class AutoDiffToy:
    """
    Attributes:
        self.val [float]: The value of the AutoDiff number
        self.der [float]: The value of the derivative of the AutoDiff number
    """
    def __init__(self, val=0.0, der=0.0):
        self.val = val
        self.der = der

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
            other (AutoDiffToy/float): The other number to subtract
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
            other (AutoDiffToy/float): The other number to subtract
        returns:
            AutoDiffToy: A new AutoDiffToy number with the delta between the numbers
        """
        try: # This should work if both are of AutoDiffToy class
            return AutoDiffToy(other.val - self.val, other.der - self.der)
        except AttributeError: # If not, we will catch the error and subtract the following way
            return AutoDiffToy(other.val - self, other.der)

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

    # Implementing the rmul in case the multiplication of a non object will be form the left
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


## TODO: Overload the division operator
## TODO: Overload the power operator
