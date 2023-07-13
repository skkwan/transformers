import math
import numpy as np
import os

def myGELU(dataIn):
    out = dataIn * 0.5 * (1 + math.erf(dataIn/math.sqrt(2)))
    print("GELU({}), erf argument is {}, output is {}".format(dataIn, math.erf(dataIn/math.sqrt(2)), out))
    return out

# def myTanhGELU(dataIn):
#     out =  0.5 * dataIn * (1 + math.tanh( np.sqrt(2.0/math.pi) * (dataIn + 0.044715 * (dataIn ** 3))) )
#     return out

# Generate a hundred outputs from -8.0 to 8.0
myInputs = np.random.uniform(-8.0, 8.0, 100)

if os.path.exists("input.txt"):
    os.system("rm input.txt")
if os.path.exists("idealOut.txt"):
    os.system("rm idealOut.txts")

for myInput in myInputs:
    with open("input.txt", "a") as fIn:
        fIn.write("{}\n".format(myInput))
    with open("idealOut.txt", "a") as fIdealOut:
        fIdealOut.write("{}\n".format(myGELU(myInput)))