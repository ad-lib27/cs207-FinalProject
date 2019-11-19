from adlib.interact import getfunc, getxvalue
import mock
import pytest

def test_bad_function():
    with mock.patch('builtins.input', return_value="3x"):
        assert getfunc() == False
    
def test_good_function():
    with mock.patch('builtins.input', return_value="3*x+5"):
        assert getfunc() == "3*x+5"

def test_bad_value():
    with mock.patch('builtins.input', return_value="hello"):
        assert getxvalue() == False

def test_good_value():
    with mock.patch('builtins.input', return_value="2"):
        assert getxvalue() == 2
