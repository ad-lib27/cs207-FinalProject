from adlib27.elem_function import exp, sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh, logistic, log, log2, log10, logb, sqrt
from adlib27.autodiff import AutoDiff as AD


def getvars():
    yeslist = ["y", "Yes", "Y", "yes"]
    more = True
    variables = []

    if input("Welcome to AdLib27. Does your function contain any variables? [y/n]: ") in yeslist:
         while more == True:
            a = input("Please enter your variable: ")
            if a.isalpha() == True and a not in variables:
                variables.append(a)
            else:
                print("Invalid or duplicate variable name. Please retry.")
            if input("Would you like to enter another variable? [y/n]: ") in yeslist:
                more = True
            else:
                more = False
    print("Your variables are: ", variables)
    return variables

def getfunc(variables):
    #dummy x-value for checking function validity
    #collects function from user input, checks for validity
    #does not check if function is composed of functions outside of milestone scope

    func = input("Enter a Python-formatted function for evaluation: ")
    dummyfunc = func
    x=0

    for var in variables:
        if not var in dummyfunc:
            print("This function is invalid or does not contain all variables, please retry.")
            return False
        else:
            dummyfunc = dummyfunc.replace(var,"x")
    try:
        exec(dummyfunc)
    except:
        print("This is not a valid function, please retry.")
        return False
    else:
        return func
    #allowing repeated attempts for x-value entry

def getpointnum():
    try:
         x = int(input("How many points would you like to evaluate at? "))
    except:
        print("This is not a valid number of evaluation points, please retry")
        return False
    else:
        return x

def getvalues(variables, pointnum):
    i = 0
    values =[]
    j = 0
    while i < len(variables):
        print("Enter an evaluation value for the variable '", variables[i], "' as a python-formatted list of length", pointnum, "of floats or integers:")
        values.append(input().strip('][').split(', '))
        i += 1
    while j < len(values):
        try:
            values[j] = list(map(float, values[j]))
            j += 1
        except:
            print("Not all evaluation values are valid, please re-start entries.")
            return False
    return values

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

def reprresult(values, variables, result, pointnum, func):
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
                pointrep.append(str(values[b][a]))
                pointsrep.append(''.join(pointrep))
                b += 1
            valrep.append(', '.join(pointsrep))
            a += 1
        return valrep

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
                ddrep.append(str(result.der[b][a]))
                drep.append(''.join(ddrep))
                b += 1
            derrep.append(', '.join(drep))
            a += 1
        return derrep

    vals = reprvals(values, variables, pointnum)
    ders = reprders(result, variables, pointnum)
    c = 0
    reslist = []

    while c < pointnum:
        resmini = []
        resmini.append("For evaluation values (")
        resmini.append(vals[c])
        resmini.append("), the value of the function f=")
        resmini.append(str(func))
        resmini.append(" is: ")
        resmini.append(str(result.val[c]))
        resmini.append(" and the the derivatives are: ")
        resmini.append(ders[c])
        reslist.append(''.join(resmini))
        c += 1

    return '\n\n'.join(reslist)

def main():
    gv = getvars()
    gf = getfunc(gv)
    while gf == False: gf = getfunc(gv)
    print("Successful function entry!")
    gp= getpointnum()
    while gp == False: gp = getpointnum()
    gx = getvalues(gv, gp)
    while gx == False: gx = getvalues(gv, gp)
    print("Successful value entry!\n")
    print(reprresult(gx,gv,getresult(gx,gv,gf),gp,gf))

if __name__ == '__main__':
    main()
