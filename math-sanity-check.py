import math
import numpy as np

def myGELU(dataIn):
    out = dataIn * 0.5 * (1 + math.erf(dataIn/math.sqrt(2)))
    print("GELU({}), erf argument is {}, output is {}".format(dataIn, math.erf(dataIn/math.sqrt(2)), out))
    return out

# def myTanhGELU(dataIn):
#     out =  0.5 * dataIn * (1 + math.tanh( np.sqrt(2.0/math.pi) * (dataIn + 0.044715 * (dataIn ** 3))) )
#     return out


myInputs = [-3.0, -1.0, 0.0, 1.0, 3.0]
myIdealOutputs = []
myOutputs = []


for myInput in myInputs:
    myIdealOutputs.append(myGELU(myInput))
    # # myOutputs.append(myTanhGELU(myInput))

print("myIdealOutputs:", myIdealOutputs)
# print("myTanhOutputs:", myOutputs)
