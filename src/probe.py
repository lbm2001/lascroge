import numpy as np

a = np.array([ [1,2,3],
              [0,2,3],
              [0,2,3] ])

mean = np.mean(a, axis=0)
std = np.std(a, axis=0)
print("Mean: ", mean)
print("Std: ", std)
