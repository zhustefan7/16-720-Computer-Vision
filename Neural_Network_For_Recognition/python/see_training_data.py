import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
import pickle



valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

test_image1 = valid_x[10,:]
test_image1 = np.reshape(test_image1,(32,32))
print(test_image1)
plt.imshow(test_image1)
plt.show()

print(valid_x.shape)