from elem_function import exp, sin, cos, tan
from autodiff import AutoDiffToy as ADT

def main():
    #dummy x-value for checking function validity 
    x=0

    #allowing repeated attempts for successful function entry
    while True:
        #collects function from user input, checks for validity
        #does not check if function is composed of functions outside of milestone scope
        func = input("Enter a Python-formatted function f(x) for evaluation:")
        try: 
            exec(func)
        except:
            print("This is not a valid function, please retry.")
        else: break
    #allowing repeated attempts for x-value entry
    while True:
        try: x = float(input("Enter an x-value to evaluate, where x is a real scalar:"))
        except ValueError:
            print("This is not a valid x-value, please retry.")
        else: break

    #prints value and derivative of result
    x = ADT(x)
    result = eval(func)
    print (f"Function Value: {result.val} Function Derivative: {result.der}")
if __name__ == '__main__':
    main()