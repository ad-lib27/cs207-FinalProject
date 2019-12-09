import math

from adlib27.autodiff import AutoDiff as AD
from adlib27.elem_function import sin

# function to perform optimization
def optimize(ad_func, domain, vector_index=0):
    # helper function to determine whether an inflection point is a max or min
    def detect_extrema(y1, y2):
        if y1 > 0 and y2 < 0:
            return "maximum"
        if y1 < 0 and y2 > 0:
            return "minimum"
        return None

    # loop through derivatives to find inflection points (looking for changes in sign)
    extrema = [{"input range": (domain[0], domain[0]), "value range": (ad_func.val[0], ad_func.val[0]), "inflection type": "endpoint"}, {"input range": (domain[-1], domain[-1]), "value range": (ad_func.val[-1], ad_func.val[-1]), "inflection type": "endpoint"}]
    for i in range(len(domain) - 1):
        inflection = detect_extrema(ad_func.der[vector_index][i], ad_func.der[vector_index][i + 1])
        if inflection:
            extrema += [{"input range": (domain[i], domain[i + 1]), "value range": (ad_func.val[i], ad_func.val[i + 1]), "inflection type": inflection}]

    # get a list of the value ranges for each inflection
    values = [d["value range"] for d in extrema]

    # return the global max and global min, as well as all the other extrema, with metadata
    results = {"global maximum": {"input range": extrema[values.index(max(values))]["input range"], "value range": max(values)}, "global minimum": {"input range": extrema[values.index(min(values))]["input range"], "value range": min(values)}, "all extrema": extrema}

    return results
