# Importing the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# The years and children count data
years = [2008, 2015, 2023]
children_in_house = [1, 2, 7]

# Transform the list of years to a 2D array for sklearn
X = np.array(years).reshape(-1, 1)
y = np.array(children_in_house)

# Transform the array to allow for polynomial regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Fit a polynomial regression model to the data
model = LinearRegression().fit(X_poly, y)

# Predict the number of children in 2030
prediction_year = np.array([2030]).reshape(-1, 1)
prediction_year_poly = poly.fit_transform(prediction_year)
prediction = model.predict(prediction_year_poly)

# Create a range of years including prediction year
years_extended = list(range(2008, 2031))

# Predict for all years
X_extended = np.array(years_extended).reshape(-1, 1)
X_extended_poly = poly.fit_transform(X_extended)
predictions = model.predict(X_extended_poly)

# Plot the data and prediction
plt.figure(figsize=(10, 6))
plt.plot(years, children_in_house, marker='o', label='Data')
plt.plot(years_extended, predictions, label='Polynomial Regression', linestyle='--')
plt.scatter(2030, prediction, color='red')  # Highlight the prediction for 2030
plt.xlabel('Year')
plt.ylabel('Number of Children in the House')
plt.title('Number of Children in the House Over Time')
plt.legend()
plt.grid(True)
plt.show()
