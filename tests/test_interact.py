from adlib27.interact import getvars, getfunc, getpointnum, getvalues, get_result, reprresult, getmode, getoptvals, repropt, contopt
from adlib27.optimize import optimize
import mock
import pytest

#tests for getting variables
def test_no_variable():
    with mock.patch('builtins.input', return_value="n"):
        assert getvars() == []

def test_two_variable():
    with mock.patch('builtins.input', side_effect = ['y', 'a', 'y', 'a', 'y', 'b', 'n']):
        vars = ['a', 'b']
        assert getvars() == vars

#test for function entered with missing variables
def test_missing_variables():
    with mock.patch('builtins.input', return_value="3*a"):
        assert getfunc(['a', 'b']) == False

#function with improper formatting
def test_bad_function():
    with mock.patch('builtins.input', return_value="3a*b"):
        assert getfunc(['a', 'b']) == False

#function with proper formatting
def test_good_function():
    with mock.patch('builtins.input', return_value="3*a*b"):
        assert getfunc(['a', 'b']) == "3*a*b"

#tests for determining mode
def test_point_mode():
    with mock.patch('builtins.input', return_value="a"):
        assert getmode() == 0

def test_opt_mode():
    with mock.patch('builtins.input', return_value="b"):
        assert getmode() == 1

def test_no_mode():
    with mock.patch('builtins.input', return_value="c"):
        assert getmode() == 2

#tests for bad and good entries for optimization values
def test_bad_optstep():
    with mock.patch('builtins.input', return_value="a"):
        assert getoptvals(['a', 'b']) == False

def test_bad_optstart():
    with mock.patch('builtins.input', side_effect=['3', 'a']):
        assert getoptvals(['a', 'b']) == False

def test_bad_optend():
    with mock.patch('builtins.input', side_effect=['3', '0', 'b']):
        assert getoptvals(['a', 'b']) == False

def test_good_optvals():
    with mock.patch('builtins.input', side_effect=['1', '0', '2']):
        assert getoptvals(['a', 'b']) == [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]

#tests for number of points for evaluation mode
def test_bad_pointnumber():
    with mock.patch('builtins.input', return_value="4.2"):
        assert getpointnum() == False

def test_good_pointnumber():
    with mock.patch('builtins.input', return_value="4"):
        assert getpointnum() == 4

#test for value entries for evaluation mode
def test_bad_value():
    with mock.patch('builtins.input', return_value="[m, 3]"):
        assert getvalues(['a'], 2) == False

def test_good_value():
    with mock.patch('builtins.input', return_value="[1, 3]"):
        assert getvalues(['a'], 2) == [[1, 3]]

#testing result getting and representing
def test_get_result():
    assert get_result([[3, 4]],'x','3*x').val == [9.0, 12.0]
    assert get_result([[3, 4]],'x','3*x').der == [[3.0, 3.0]]

def test_reprresult():
    assert reprresult([[3.0, 4.0]], ['x'], get_result([[3.0, 4.0]],'x','3*x'), 2, '3*x') == "For evaluation values (x = 3.00000), the value of the function f=3*x is: 9.00000 and the the derivatives are: df/dx = 3.00000\n\nFor evaluation values (x = 4.00000), the value of the function f=3*x is: 12.00000 and the the derivatives are: df/dx = 3.00000"

def test_conopt_yes():
    with mock.patch('builtins.input', return_value="y"):
        assert contopt() == True

def test_conopt_no():
    with mock.patch('builtins.input', return_value="n"):
        assert contopt() == False

def test_repropt():
    assert repropt(optimize([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['x'], '3*x'), [[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]], ['x'], '3*x') == "\nIn the domain 0.00000 to 1.00000, the extrema for 3*x represented as ['x'] are:\nThe local minimum is the endpoint located in the range ([0.0], [0.0]) valued in the range (0.0, 0.0)\nThe local maximum is the endpoint located in the range ([1.0], [1.0]) valued in the range (3.0, 3.0)"
