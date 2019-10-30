# Milestone 1 Document

## Table of Contents
- [Introduction](#introduction)
- [Background](#background)
- [How to Use ad-lib27](#howtouse)
  - [Examples](#examples)
- [Software Organization](#software_org)
  - [Directory Structure](#structure)
  - [Included Modules](#modules)
  - [Tests](#tests)
  - [Package and Framework](#package)
  - [Distribution](#dist)
- [Implementation](#implementation)
  - [Data Structures](#data_structures)
  - [Elem_function](#elem_function)
  - [Input_parser](#input_parser)
  - [External Dependencies](#dependencies)


## Introduction <a name="introduction"></a>

Automatic differentiation is a method to evaluate derivatives of any linear or nonlinear systems that differs from other traditional methods such as symbolic differentiation or numeric differentiation. Numeric differentiation can be simple to implement but it can also introduce rounding and truncation errors and thus make the results inaccurate. Symbolic differentiation can be time consuming and also result in complex and cryptic expressions. Furthermore, Symbolic methods require the model to be close-form which are not applicable in many machine learning problems.

Automatic differentiation address above issues by using chain rules and computing derivatives up to machine precision. It has been well established in many fields including atmospheric sciences, fluid dynamics, optimization, and has been rapidly growing in machine learning.


## Background <a name="background"></a>

As explained in the introduction, automatic differentiation evades both the approximation problems of numeric differentiation and the complexity problems of symbolic differentiation. Essentially, automatic differentiation continuously applies the chain rule to a function at a certain point. Automatic differentiation is able to do this by exploiting the fact that all functions input by a user to a computer can be represented as some combination of elementary functions. These elementary functions include arithmetic functions, trigonometric functions, exponential functions, logarithmic functions, and nth root functions. The function is expanded in terms of these elementary functions, and the chain rule is applied by writing an array of derivatives, each in terms of the last, until a final derivative is found. The implementation of this method is further explained below.

For forward automatic differentiation, first, the function is rewritten as a map of elementary functions applied to the input, with each step in the map being represented by a distinct trace. For example, if the function is “5x + 1”, some input trace x1 becomes trace x2 by being multiplied by 5, which becomes trace x3 by adding 1, and then x3 is the output. Every elementary function applied to the input is represented by such a change in the trace. This mapping is represented as a flowchart between traces, with elementary functions represented by arrows between traces.

This map is used to generate the elementary function, elementary function value, elementary function derivative, and derivative value for each trace. In the case of multi-value inputs to the function (for example, “5x +3y +1”), a partial derivative is taken with respect to each input. For each trace, the elementary function and elementary function derivative are written in terms of the previously-evaluated, constituent traces, and evaluated at the values and derivative values of these constituent traces. In this way, the traces are recursively determined through a continuous application of the chain rule. The evaluated derivative of the final trace becomes the output, and forward automatic differentiation is complete.


## How to Use ad-lib27 <a name="howtouse"></a>

We envision that the end user will install our package in their own terminal using the pip install command.

```
pip install ad-lib27
```

After installing our package, the user can import it into their own program(s) as demonstrated in the following examples. Alternatively, should the user elect to clone the GitHub repository directly into their own workspace, the user can also run our interact.py file where, via the terminal interface, the user will be prompted to provide a function as input, The function can be a scalar constant, a trigonometric functions such as sine, cosine, etc., it can be exponential, and even multivariable (more details on supported functions in the implementation section). The second and third inputs for which the user will be prompted are the values (of `x` and `y`) at which to evaluate the function and its derivative.

As mentioned in the first sentence of the above paragraph, if the user is a developer who would want to use our package in his/her code, he/she will need to import it the following way:

```
>>> from ad-lib27 import autodifff as AD
```

The user would be able to create an object by passing a function as a string to the autodifff constructor. Then, in order to calculate the value and the derivative of the function at a certain value, the user will need to use the `get_der` and `get_val` methods.

### Examples <a name="examples"></a>

Scalar function
```
my_func = AD.Elem_function(x^2+1)
my_func.get_val(3) → will produce the output: 10
my_func.get_der(3) → will produce the output: 6
```

Trigonometric functions (`sin x`, `cos x`, `tan x`)
```
my_func = AD.Elem_function(sin(x))
my_func.get_val(0) → will produce the output: 0
my_func.get_der(0) → will produce the output: 1
```

Resetting a function value
```
my_func = AD.Elem_function(2)
my_func.set_val(x^2 + sin(y))
```

## Software Organization <a name="software_org"></a>

### Directory Structure <a name="structure"></a>
```
  cs207-FinalProject/
      ad-lib27/
          __init__.py
          autodifff.py
          interact.py
          Input_parser.py
          Elem_function.py
      tests/
      docs/
      setup.py
```
### Included Modules <a name="modules"></a>
  - \_\_init\_\_.py

    This module is used for package initialization.

  - autodifff.py

    This module applies forward mode of automatic differentiation to the input function using chain rule.

  - interact.py

    This module takes the input variable values and input function, and feeds it into Input_parser. It also outputs the trace table and the variable values when the derivative of input function equals to 0.

  - Input_parser.py

    This module parses and identify the input function and return the corresponding derivatives to the elementary function.

  - Elem_function.py

    This module evaluates derivatives of various elementary functions.

### Tests <a name="tests"></a>
  - Tests will be in a tests/ directory.
  - External continuous integration service such as TravisCI will be used for testing. CodeCov will be used to ensure the coverage of testing.
  - Both TravisCI and CodeCov badges are embedded in README.md.

### Package and Framework <a name="package"></a>

  - We will first add a setup.py module in the root directory and use ```python setup.py sdist```to package our software, then use ```python setup.py register``` to register our package on PyPI.
  - Frameworks are not needed here since our software packaging is pretty straightforward and does not depend on any external service.

### Distribution <a name="dist"></a>

  The package will be available on PyPI and users can install the package using ```pip install ad-lib27```.


## Implementation <a name="implementation"></a>

### Data Structures <a name="data_structures"></a>

The core data structures used in our implementation will be classes (e.g., to represent our supported elementary functions), lists (e.g., for our trace evaluation steps), and dictionaries (e.g., for our function/derivative lookup functionality).

### Elem_function <a name="elem_function"></a>

In particular, as referred to above, we’ll need to define an `Elem_function` class to be able to support the following elementary functions:
- Constant functions (`2`, `pi`, etc)
- Powers of x (`x`, `x^2`, `x^3`, etc)
- Roots of x, (`sqrt(x)`)
- Exponential functions (`e^x`)
- Logarithms (`log x`)
- Trigonometric functions (`sin x`, `cos x`, `tan x`)
- Inverse trigonometric functions (`arcsin x`, `arccos x`, `arctan x`)
- Hyperbolic functions (`sinh x`, `cosh x`, `tanh x`)
- All functions obtained by adding, subtracting, multiplying, dividing, and composing any of the above functions
Unsurpringly, the `Elem_function` class would need to include the `self.val` and `self.der` attributes (with corresponding getter methods, as well as a setter method for `self.val`), and overloaded `__add__`, `_radd__`, `__mul__`, and `__rmul__` dunder methods.

### Input_parser <a name="input_parser"></a>

We’ll also need to define an `Input_parser` class to identify the function specified by the user in `interact.py`. This class should have a `parse` method to, given a string representation of a function, parse the input into the corresponding elementary function. We will need to write clear documentation for this feature such that users can reference it to input properly formatted string representations of functions.
exam
In terms of parsing input, we can construct a dictionary of supported elementary functions (e.g., `sin`, `sqrt`, `log`, `exp`, etc.) with their corresponding derivatives, such that parsing a user's input is as simple as mapping substrings of their given function string. We can then construct the corresponding `Elem_function` objects underneath the hood.

### External Dependencies <a name="dependencies"></a>

We anticipate that we'll use the numpy library to support our math operations, though we're unsure at the moment of which other external dependencies we may rely on (we'll likely discover this as we begin to actually work on the implementation).
