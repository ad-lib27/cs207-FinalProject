from adlib27.interact import getvars, getfunc, getpointnum, getvalues, get_result, reprresult
import mock
import pytest

def test_no_variable():
    with mock.patch('builtins.input', return_value="n"):
        assert getvars() == []

def test_two_variable():
    with mock.patch('builtins.input', side_effect = ['y', 'a', 'y', 'a', 'y', 'b', 'n']):
        vars = ['a', 'b']
        assert getvars() == vars

def test_missing_variables():
    with mock.patch('builtins.input', return_value="3*a"):
        assert getfunc(['a', 'b']) == False

def test_bad_function():
    with mock.patch('builtins.input', return_value="3a*b"):
        assert getfunc(['a', 'b']) == False

def test_good_function():
    with mock.patch('builtins.input', return_value="3*a*b"):
        assert getfunc(['a', 'b']) == "3*a*b"

def test_bad_pointnumber():
    with mock.patch('builtins.input', return_value="4.2"):
        assert getpointnum() == False

def test_good_pointnumber():
    with mock.patch('builtins.input', return_value="4"):
        assert getpointnum() == 4

def test_bad_value():
    with mock.patch('builtins.input', return_value="[m, 3]"):
        assert getvalues(['a'], 2) == False

def test_good_value():
    with mock.patch('builtins.input', return_value="[1, 3]"):
        assert getvalues(['a'], 2) == [[1, 3]]

def test_get_result():
    assert get_result([[3, 4]],'x','3*x').val == [9.0, 12.0]
    assert get_result([[3, 4]],'x','3*x').der == [[3.0, 3.0]]

def test_reprresult():
    assert reprresult([[3.0, 4.0]], ['x'], get_result([[3.0, 4.0]],'x','3*x'), 2, '3*x') == "For evaluation values (x = 3.0), the value of the function f=3*x is: 9.0 and the the derivatives are: df/dx = 3.0\n\nFor evaluation values (x = 4.0), the value of the function f=3*x is: 12.0 and the the derivatives are: df/dx = 3.0"
