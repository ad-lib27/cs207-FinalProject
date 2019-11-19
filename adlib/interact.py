from adlib.elem_function import exp, sin, cos, tan
from adlib.autodiff import AutoDiffToy as ADT

def getfunc():
    #dummy x-value for checking function validity 
    x=0
    #collects function from user input, checks for validity
    #does not check if function is composed of functions outside of milestone scope
    func = input("Enter a Python-formatted function f(x) for evaluation:")
    try: 
        exec(func)
    except:
        print("This is not a valid function, please retry.")
        return False
    else: 
        return func
    #allowing repeated attempts for x-value entry

def getxvalue():
    try: x = float(input("Enter an x-value to evaluate, where x is a real scalar:"))
    except ValueError:
            print("This is not a valid x-value, please retry.")
            return False
    else:
        return x

def main():
    gf = getfunc()
    while gf == False:
        gf = getfunc()
    print("Successful function entry!")
    gx = getxvalue()
    while gx == False:
        gx = getxvalue()
    print("Successful x-value entry!")
    x = ADT(gx)
    result = eval(gf)
    print(f"Function Value: {result.val} Function Derivative: {result.der}")      

if __name__ == '__main__':
    main()