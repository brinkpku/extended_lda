# !/usr/bin/env python3

import os
import time

import configs

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

def clear():
    if os.path.exists(configs.NEWSINPUT):
        os.remove(configs.NEWSINPUT)
        print("removed", configs.NEWSINPUT)
    if os.path.exists(configs.ABSTRACTINPUT):
        os.remove(configs.ABSTRACTINPUT)
        print("removed", configs.ABSTRACTINPUT)


# reltion 
DEP2STR = "{} - [{}] - {}" # governor - dep relation - dependent
DEPENDENT = "dependent"
DEPENDENTGLOSS = "dependentGloss"
GOVERNOR = "governor"
GOVERNORGLOSS = "governorGloss"
DEP = "dep"

if __name__=="__main__":
    clear()