# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 19:31:00 2020

@author: ASHLIN GABRIEL

Credits to Lazy programmer training videos at udemy
"""


#Matplotlib Line charts

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,20,1000)
y = np.sin(x) + 0.2 * x
plt.xlabel('input')
plt.ylabel('output')
plt.title("My Plot")
plt.plot(x,y)
plt.show()

#Scatter plot

X = np.random.randn(100,2)
plt.scatter(X[:,0],X[:,1])
plt.xlabel('input')
plt.ylabel('output')
plt.title("My Plot")
plt.show()

X = np.random.randn(200,2)
X[:50] += 3 # I say from index 0 fetch 50 rows and add it by 3
Y= np.zeros(200) 
Y[:50] = 1
plt.scatter(X[:,0],X[:,1], c = Y)


#Histogram 

X = np.random.randn(10000)
plt.hist(X)
plt.hist(X,bins=50)

X= np.random.random(10000) #Uniform distributed
plt.hist(X)
plt.hist(X,bins=50)

#Plotting images

from PIL  import Image

im = Image.open("lena.png")
type(im)

arr = np.array(im)

arr.shape

plt.imshow(arr);
plt.imshow(im);


gray = arr.mean(axis=2)
gray.shape

plt.imshow(gray)

plt.imshow(gray, cmap='pink');
