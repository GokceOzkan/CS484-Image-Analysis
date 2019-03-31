#dddgggg
#dddd

import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as img

default_filter = np.random.randint(10, size=(2,2))

def check_index(index):
    if index < 0:
        return 0
    else:
        return index


def convolution(original_image = None, filter=default_filter):
    #original_image = img.imread(original_image)
    original_image = np.asarray(original_image)
    filter = np.asarray(filter)

    #filter_height, filter_width = filter.shape
    #original_image_height, original_image_width = original_image.shape


    #new_image = np.zeros(check_index(original_image.shape[0] - filter.shape[0] + 1), check_index(original_image.shape[1] - filter.shape[1] + 1))
    #xx = original_image.shape[0]
    #yy = original_image.shape[1]
    #aa, bb = kernel.shape
    #new_image = np.zeros((xx,yy))
    new_image = np.ones((original_image.shape[0] - filter.shape[0] + 1, original_image.shape[1] - filter.shape[1] + 1))


    for i in range(original_image.shape[0] - filter.shape[0]): #height
        for j in range(original_image.shape[1] - filter.shape[1]): #width
            #top = i
            #bottom = i + (filter.shape[0])
            #left = j
            #right = j + (filter.shape[1])
            #overlap = original_image[check_index(top):int(bottom), check_index(left):int(right)]
            #overlap = original_image[i:i + filter.shape[0] , j:j + filter.shape[1]]
            overlap = original_image[check_index(i):int(i+filter.shape[0]), check_index(j):int(j+filter.shape[1])]

            sum = 0

            for a in range(filter.shape[0]):
               for b in range(filter.shape[1]):
                   # sum = sum + overlap[a:a+1 , b:b+1]*filter[a:a+1 , b:b+1]
                    #print(overlap[a][b])
                    #print("pass")

            #new_image[i:i+1, j:j+1] = sum
                    new_image[i,j] = sum
        return new_image
        #deneme=3


img = cv2.imread("sonnet.png")
img = np.uint8(img)
myfilter = np.random.randint(10, size=(2,2))
imRes = convolution(img, myfilter)
imRes = np.uint8(imRes)
#cv2.imshow("original", img)
#cv2.imshow("convolved", imRes)

cv2.waitKey(0)
cv2.destroyAllWindows()
