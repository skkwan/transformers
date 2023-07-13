import numpy as np
import matplotlib.pyplot as plt

x = [6, 7, 8, 9, 10]
y = [0.00706373, 0.00693015, 0.00685544, 0.00676582, 0.00811899]

plt.title("Root Mean Square (RMS) error")
plt.plot(x, y, '-o', color = 'red', label = "LUT, ap_fixed<18,8>")

plt.xlabel('ap_fixed<16, x> of input and outputs')
plt.ylabel('RMS error')

plt.legend()

plt.savefig("rmsError.png")
