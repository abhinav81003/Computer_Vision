import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from public_tests import *

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2])));

x = np.array([0.5, 0, 2.0])
output = sigmoid(x)
print(output)