# !/usr/bin/env python3

import time

def timer(func):
    """
    log function exec time
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("func {} exec time: {}".format(func.__name__, end - start))
        return result
    return wrapper


# reltion 
DEP2STR = "{} - [{}] - {}" # governor - dep relation - dependent
DEPENDENT = "dependent"
DEPENDENTGLOSS = "dependentGloss"
GOVERNOR = "governor"
GOVERNORGLOSS = "governorGloss"
DEP = "dep"