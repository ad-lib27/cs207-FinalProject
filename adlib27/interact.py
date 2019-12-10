from adlib27.optimize import optimize
from adlib27.elem_function import exp, sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh, logistic, log, log2, log10, logb, sqrt
from adlib27.autodiff import AutoDiff as AD
import numpy as np

#initialize the program and ask for variables
def getvars():
    yeslist = ["y", "Yes", "Y", "yes"]
    more = True
    variables = []

    if input("Welcome to AdLib27. Does your function contain any variables? [y/n]: ") in yeslist:
         while more == True:
            #if there are variables, must enter a variable that is composed of letters
            a = input("Please enter your variable: ")
            if a.isalpha() == True and a not in variables:
                variables.append(a)
            #if variable already exists or is not a letter, retry
            else:
                print("Invalid or duplicate variable name. Please retry.")
            #ask if user wants to enter another function, then execute loop again if true.
            if input("Would you like to enter another variable? [y/n]: ") in yeslist:
                more = True
            else:
                more = False
    #return collected variables
    print("Your variables are: ", variables)             
    return variables

#function to get the function that will be evaulated
def getfunc(variables):
    #ask for a formatted function (using the formatting in out documentation, and standard to python)
    func = input("Enter a Python-formatted function for evaluation: ")
    dummyfunc = func
    x=1

    #check if function contains all variables
    for var in variables:
        if not var in dummyfunc:
            print("This function is invalid or does not contain all variables, please retry.")
            return False
        else:
            dummyfunc = dummyfunc.replace(var,"x")
    try:
        #check if function is executable, using a dummy variable x
        exec(dummyfunc)
    except:
        #if not valid, retry entry
        print("This is not a valid function, please retry.")
        return False
    else: 
        return func

#function to get the mode: evaluating at a given point, or optimizing over a range
def getmode():
    mode = input("Would you like to \n(a) evaluate at a point, or \n(b) optimize over a range? [a/b] ")
    if mode.lower() == 'a':
        return 0
    elif mode.lower() == 'b': 
        return 1
    else: 
        return 2

#function to get values for optimization
def getoptvals(variables):
    #ask for some step size that must be a float
    step = input("What step size would you like to take while finding values? Please enter a float. Note that small step sizes will increase program runtime. ")
    try: 
        step = abs(float(step))
    except:
        #if not a float, retry
        print("Not a valid step size")
        return False
    #ask for start of range that must be float
    start = input("At what value would you like to start searching? ")
    try:
        start = float(start)
    except:
        print("Not a valid start point")
        return False 
    #ask for end of range that must be floats
    end = input("At what value would you like to stop searching? ")
    try:
        end = float(end)
        assert end > start
    except:
        print("Not a valid end point")
        return False
    #create list of values to test using numpy linspace
    optval = list(np.linspace(start, end, ((end-start)/step)+1))
    
    return optval

#for point evaluation mode, find number of points being evaluated
def getpointnum():
    x = input("How many points would you like to evaluate at? ")
    try:
         x = abs(int(x))
    except:
        print("This is not a valid number of evaluation points, please retry")
        return False
    else: 
        return x

#for point evaluation mode, ask for places to evaluate for each variables
def getvalues(variables, pointnum):
    i = 0
    values =[]
    j = 0
    while i < len(variables):
        print("Enter an evaluation value for the variable '", variables[i], "' as a python-formatted list of length", pointnum, "of floats or integers:")
        values.append(input().strip('][').split(', '))
        i += 1
    #map to floats, must be valid
    while j < len(values):
        try: 
            values[j] = list(map(float, values[j]))
            j += 1
        except: 
            print("Not all evaluation values are valid, please re-start entries.")
            return False
    return values

#use ADlist to evaluate value and derivative for a function 
def getresult(values,variables,func):
    funcdict = {"exp": exp, "sin": sin, "cos": cos, "tan": tan, "arcsin": arcsin, "arccos": arccos, "arctan": arctan, "sinh": sinh, "cosh": cosh, "tanh": tanh, "logistic": logistic, "log": log, "log2": log2, "log10": log10, "logb": logb, "sqrt": sqrt}
    i = 0
    ADlist = []
    vardict = {}
    while i < len(variables):
        ADlist.append(AD(values[i],i,len(variables)))
        vardict[variables[i]] = ADlist[i]
        i += 1
    return eval(func,funcdict,vardict)

#represent result for point evaluation mode
def reprresult(values, variables, result, pointnum, func):
    #represent values after evaluation
    def reprvals(values, variables, pointnum):
        valrep = []
        a = 0
        while a < pointnum:
            b = 0
            pointsrep = [] 
            while b < len(variables):
                pointrep = []
                pointrep.append(str(variables[b]))
                pointrep.append(" = ")
                pointrep.append("{:.5f}".format(values[b][a]))
                pointsrep.append(''.join(pointrep))
                b += 1
            valrep.append(', '.join(pointsrep))
            a += 1
        return valrep

    #represent derivatives after evaluation
    def reprders(result, variables, pointnum):
        derrep = []
        a = 0
        while a < pointnum:
            b = 0
            drep = [] 
            while b < len(variables):
                ddrep = []
                ddrep.append("df/d")
                ddrep.append(str(variables[b]))
                ddrep.append(" = ")
                ddrep.append("{:.5f}".format(result.der[b][a]))
                drep.append(''.join(ddrep))
                b += 1
            derrep.append(', '.join(drep))
            a += 1
        return derrep

    vals = reprvals(values, variables, pointnum)
    ders = reprders(result, variables, pointnum)
    c = 0  
    reslist = []

    #put all together into string for output, for each point
    while c < pointnum:
        resmini = []
        resmini.append("For evaluation values (")
        resmini.append(vals[c])
        resmini.append("), the value of the function f=")
        resmini.append(str(func))
        resmini.append(" is: ")
        resmini.append("{:.5f}".format(result.val[c]))
        resmini.append(" and the the derivatives are: ")
        resmini.append(ders[c])
        reslist.append(''.join(resmini))
        c += 1

    return '\n\n'.join(reslist)

#represent results for optimization
def repropt(optimized, vals, variables, func):

    #within a domain, represent local min and max for function
    optout = []
    optout.append("\nIn the domain ")
    optout.append("{:.5f}".format(vals[0]))
    optout.append(" to ")
    optout.append("{:.5f}".format(vals[-1]))
    optout.append(", the extrema for ")
    optout.append(str(func))
    optout.append(" represented as ") 
    optout.append(str(variables))
    optout.append(" are:")
    optout.append("\nThe local minimum is the ")
    optout.append((str(optimized.get("global minimum").get("inflection type"))))
    optout.append(" located in the range ")
    optout.append(str(optimized.get("global minimum").get("input range")))
    optout.append(" valued near ")
    optout.append(str(optimized.get("global minimum").get("value range")))
    optout.append("\nThe local maximum is the ")
    optout.append((str(optimized.get("global maximum").get("inflection type"))))
    optout.append(" located in the range ")
    optout.append(str(optimized.get("global maximum").get("input range")))
    optout.append(" valued near ")
    optout.append(str(optimized.get("global maximum").get("value range")))
    
    return ''.join(optout)
        
#ask if cotinued optimizing after output
def contopt():
    yeslist = ["y", "Yes", "Y", "yes"]
    if input("Would you like to continue optimizing this function? [y/n] ") in yeslist:
        return True
    else:
        return False

def main():
    gv = getvars()
    gf = getfunc(gv)
    while gf == False: gf = getfunc(gv)
    gm = getmode()
    while gm == 2: gm = getmode()
    if gm == 0:
        gp= getpointnum()
        while gp == False: gp = getpointnum()
        gx = getvalues(gv, gp)
        while gx == False: gx = getvalues(gv, gp)
        print(reprresult(gx,gv,getresult(gx,gv,gf),gp,gf))
    else:
        searching = True
        while searching == True: 
            go = getoptvals(gv)
            while go == False: go = getoptvals(gv)
            print(repropt(optimize(go,gv,gf), go, gv, gf))
            searching = contopt()      

if __name__ == '__main__':
    main()