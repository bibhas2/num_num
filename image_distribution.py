#This shows the distribution of letters in generated images
#using a heatmap. 

from image_loader import gen_images
import numpy as np
import matplotlib.pyplot as plt

test_X, test_Y = gen_images(100)

NUM_IMAGES = test_X.shape[0]

np.divide(test_X, NUM_IMAGES)

averageData = np.sum(test_X, 0)

averageData = np.reshape(averageData, (test_X.shape[1], test_X.shape[2]))

print averageData.shape

plt.imshow(averageData)
plt.show()
