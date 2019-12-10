import math
from itertools import product as prod
from adlib27.autodiff import AutoDiff as AD
from adlib27.elem_function import exp, sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh, logistic, log, log2, log10, logb, sqrt


# function to perform optimization
# domain is a list of evenly spaced, sequential floats
# variables are a list of strings
# func is a python-formatted function as a string
def optimize(domain, variables, func):

    # helper function to determine whether an inflection point is a critical point
    def detect_critpoint(y1, y2):
        if y1 > 0 and y2 <= 0:
            return "critical point"
        if y1 < 0 and y2 >= 0:
            return "critical point"
        return None

    #function to render AD objects for different trials of point values
    def get_result(values, variables, func):
        funcdict = {"exp": exp, "sin": sin, "cos": cos, "tan": tan, "arcsin": arcsin, "arccos": arccos, "arctan": arctan, "sinh": sinh, "cosh": cosh, "tanh": tanh, "logistic": logistic, "log": log, "log2": log2, "log10": log10, "logb": logb, "sqrt": sqrt}
        i = 0
        ADlist = []
        vardict = {}
        while i < len(variables):
            ADlist.append(AD(values[i],i,len(variables)))
            vardict[variables[i]] = ADlist[i]
            i += 1
        return eval(func,funcdict,vardict)

    #function to get all trial values by rotating a list to get all permutations with replacement
    def vals_generator(domain, variables):
        def rotator(listed):
            rotations = [listed]
            i = 0
            while i < len(listed)-1:
                listed = listed[1:] + listed[:1]
                rotations.append(listed)
                i += 1
            return rotations

        combos = prod(rotator(domain), repeat = len(variables))
        combolist = [list(ele) for ele in combos]

        return combolist



    # create first AD object before looping
    vals = vals_generator(domain, variables)
    k=0
    ad_func = get_result(vals[k], variables, func)

    #helper function to find [x, y, z, etc.] point after evaluation
    def getpt(vals, ind, index):
        a = 0
        pt = []
        while a < len(vals[ind]):
            pt.append(vals[ind][a][index])
            a += 1
        return pt

    #helper function to find point following a point
    def getpt2(point, step):
        p2 = [x + step for x in point]
        return p2

    #enter endpoints as critical points
    critpoints = [{"variables": variables, "input range": (getpt(vals, k, 0), getpt(vals, k, 0)), "value range": (ad_func.val[0], ad_func.val[0]), "inflection type": "endpoint"}, {"input range": (getpt(vals, k, -1), getpt(vals, k, -1)), "value range": (ad_func.val[-1], ad_func.val[-1]), "inflection type": "endpoint"}]

    #for all permutations of the given domain (rotating the list for each variable)
    while k < len(vals):
        #create AD object
        ad_func = get_result(vals[k], variables, func)
        i = 0
        #for each variable
        while i < len(domain)-1:
            done = False
            #if there is critical point for the first variable, keep looking
            inflection = detect_critpoint(ad_func.der[0][i], ad_func.der[0][i + 1])
            if inflection:
                #at the index where the critical point, make sure that all other variables have a critical point at this index
                j = 1
                while j < len(variables):
                    inflection = detect_critpoint(ad_func.der[j][i], ad_func.der[j][i + 1])
                    #no inflection, look at next variable
                    if not inflection:
                        done = True
                        break
                    #if inflection, keep looking
                    elif inflection and j != len(variables)-1:
                        j += 1
                        continue
                    #if done looking, add critical point to list
                    elif inflection and j == len(variables)-1:
                        pt = getpt(vals, k, i)
                        step = abs(domain[1]-domain[0])
                        critpoints += [{"variables": variables, "input range": (pt, getpt2(pt, step)), "value range": (ad_func.val[i], ad_func.val[i + 1]), "inflection type": inflection}]
                        done = True
                        break
                    else:
                        break
            if done == True:
                break
            #continue if no critical point for variable
            else:
                i += 1
        k += 1

    #check for duplicate critical points from search, delete if so
    def point_compare(points):
        lenp = len(points[0]["variables"])
        a = 0
        while a < len(points)-1:
            b=0
            if points[a]["input range"][0] == points[a]["input range"][1] and points[a]["inflection type"] == "critical point":
                        del points[a]
            while b < lenp:
                if points[a]["input range"][0][b] == points[a+1]["input range"][0][b]:
                    b += 1
                    if b == lenp and points[a]["inflection type"] == points[a+1]["inflection type"]:
                        del points[a]
                        break
                    else:
                        continue
                else:
                    break
            a += 1
        return points

    critpoints = point_compare(critpoints)
    # get a list of the value ranges for each inflection
    values = [d["value range"] for d in critpoints]

    # return the global max and global min, as well as all the other critpoints, with metadata
    results = {"global maximum": {"input range": critpoints[values.index(max(values))]["input range"], "value range": max(values), "inflection type": critpoints[values.index(max(values))]["inflection type"]}, "global minimum": {"input range": critpoints[values.index(min(values))]["input range"], "value range": min(values), "inflection type": critpoints[values.index(min(values))]["inflection type"]}, "all critical points": critpoints}

    return results
