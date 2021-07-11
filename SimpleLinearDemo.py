from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

# Set up the Linear Regression model
model = LinearRegression()

# Train the model with our data
model.fit(penguin_data[['Height']], penguin_data['Weight'])

# Plot our original training data
axes = plt.axes()
axes.scatter(x=penguin_data['Height'], y=penguin_data['Weight'])

# Determine the best fit line
slope = model.coef_[0]
intercept = model.intercept_

# Plot our model line
x = np.linspace(10,20)
y = slope*x+intercept
axes.plot(x, y, 'r')

# Add some labels to the graph
axes.set_xlabel('Height')
axes.set_ylabel('Weight')

plt.show()
