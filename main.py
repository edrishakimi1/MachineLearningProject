#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix  # evaluation metrics

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures

# In[ ]:


df = pd.read_csv('cars_raw.csv')
# df.head()
# df.info()

df = df[df['Price'] != 'Not Priced']
df['Price'] = df['Price'].str.replace('$', '').str.replace(',', '')
df['Price'] = df['Price'].astype(float)
df = df[df['Price'] < 60000]
df = df[df['Price'] > 10000]

# In[ ]:


df = df.drop(['Used/New', 'ConsumerRating', 'ConsumerReviews', 'SellerType', 'SellerRating',
              'SellerReviews', 'ComfortRating', 'InteriorDesignRating', 'ExteriorStylingRating', 'ReliabilityRating',
              'MinMPG', 'MaxMPG',
              'Transmission', 'PerformanceRating', 'Make', 'FuelType', 'Drivetrain'], axis=1)

df = df.drop(['SellerName', 'StreetName',
              'State', 'Zipcode', 'DealType', 'ExteriorColor', 'InteriorColor',
              'Engine', 'VIN', 'Stock#', 'ValueForMoneyRating', 'Model'], axis=1)

# In[ ]:


sn.set(rc={'figure.figsize': (20, 12)})
sn.heatmap(df.corr(), annot=True, fmt='.2g', cmap='coolwarm', vmin=-1, vmax=1)
plt.show()

# In[ ]:


pair = sn.pairplot(df)
plt.show()

# In[ ]:


y = df['Price']
X = df.drop('Price', axis=1)

# In[ ]:


# Now we choosing our predition
X_train, X_ram, y_train, y_ram = train_test_split(X, y, test_size=0.6)
X_val, X_test, y_val, y_test = train_test_split(X_ram, y_ram, test_size=0.2)

# In[ ]:


from sklearn.neural_network import MLPRegressor

num_layers = [1, 2, 4, 6, 8, 10]  # number of hidden layers
num_neurons = 15  # number of neurons in each layer
mlp_tr_errors = []
mlp_val_errors = []
linear_val_errors = []
for i, num in enumerate(num_layers):
    hidden_layer_sizes = tuple([num_neurons] * num)  # size (num of neurons) of each layer stacked in a tuple

    mlp_regr = MLPRegressor(hidden_layer_sizes, random_state=42, max_iter=1000)  # Initialise an MLPRegressor

    mlp_regr.fit(X_train, y_train)  # Train MLP on the training set

    # YOUR CODE HERE
    # raise NotImplementedError()
    X_train_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)
    X_val_poly = poly.fit_transform(X_val)
    lin_regr.fit(X_train_train_poly, y_train)

    ## evaluate the trained MLP on both training set and validation set
    y_pred_train = mlp_regr.predict(X_train)  # predict on the training set
    tr_error = mean_squared_error(y_train, y_pred_train)  # calculate the training error
    y_pred_val = mlp_regr.predict(X_val)  # predict values for the validation data
    val_error = mean_squared_error(y_val, y_pred_val)  # calculate the validation error
    y_pred_train_poly = lin_regr.predict(X_train_train_poly)  # calculate the␣ ↪training error
    y_pred_test_poly = lin_regr.predict(
        X_test_poly)  # calculate the test error y_pred_val_poly = lin_regr.predict(X_val_poly) #calculate the validation error
    y_pred_val_poly = lin_regr.predict(X_val_poly)  # calculate the validation error

    mlp_tr_errors.append(tr_error)
    mlp_val_errors.append(val_error)
    linear_val_errors.append(val_error)
    accuracy_test_poly = round(r2_score(y_test, y_pred_test_poly), 5)  # accuracy for the test
    accuracy_train_poly = round(r2_score(y_train, y_pred_train_poly), 5)  # accuracy for the training
    accuracy_val_poly = round(r2_score(y_val, y_pred_val_poly), 5)  # accuracy for the validation
print("accuracy of test  MLPRegressor: ", accuracy_test_poly)  # printing the accuracy of test  MLPRegressor
print("accuracy of training  MLPRegressor: ", accuracy_train_poly)  # printing the accuracy of training  MLPRegressor
print("accuracy of validatio  MLPRegressor: ", accuracy_val_poly)  # printing the accuracy of validation  MLPRegressor

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].set_title('Residual Plot of Train samples')
sn.distplot((y_train - y_pred_train_poly), hist=False, ax=ax[0])
ax[0].set_xlabel('y_train - y_pred_train')

# Y_test vs Y_train scatter plot

ax[1].set_title('y_test vs y_pred_test')
ax[1].scatter(x=y_test, y=y_pred_test_poly)
ax[1].set_xlabel('y_test')
ax[1].set_ylabel('y_pred_test')
plt.show()

# In[ ]:


# In[ ]:


degrees = [1, 2, 3, 4, 5]

# we will use this variables to store the resulting training and validation errors for each polynomial degree
linear_tr_errors = []
linear_val_errors = []
linear_test_errors = []
for degree in degrees:
    lin_regr = LinearRegression(fit_intercept=False)
    poly = PolynomialFeatures(degree=degree)  # generate polynomial features
    X_train_poly = poly.fit_transform(X_train)  # fit the raw features
    lin_regr.fit(X_train_poly, y_train)  # apply linear regression to these new features and labels

    y_pred_train = lin_regr.predict(X_train_poly)  # predict using the linear model
    tr_error = mean_squared_error(y_train, y_pred_train)  # calculate the training error
    X_val_poly = poly.transform(X_val)  # transform the raw features for the validation data
    y_pred_val = lin_regr.predict(X_val_poly)  # predict values for the validation data using the linear model
    val_error = mean_squared_error(y_val, y_pred_val)  # calculate the validation error
    test_error = mean_squared_error(y_test, y_pred_test_poly)  # test error

    linear_tr_errors.append(tr_error)
    linear_val_errors.append(val_error)
    linear_test_errors.append(test_error)

    accuracy_test_poly = round(r2_score(y_test, y_pred_test_poly), 5)  # accuracy for the test
    accuracy_train_poly = round(r2_score(y_train, y_pred_train_poly), 5)  # accuracy for the training
    accuracy_val_poly = round(r2_score(y_val, y_pred_val_poly), 5)  # accuracy for the validation
    test_error = mean_squared_error(y_test, y_pred_test_poly)

print("accuracy of test PolyReg : ", accuracy_test_poly)  # printing the accuracy of test PolyReg
print("accuracy of training PolyReg : ", accuracy_train_poly)  # printing the accuracy of training PolyReg
print("accuracy of validatio PolyReg : ", accuracy_val_poly)  # printing the accuracy of validation PolyReg

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].set_title('Residual Plot of Train samples')
sn.distplot((y_train - y_pred_train_poly), hist=False, ax=ax[0])
ax[0].set_xlabel('y_train - y_pred_train')

# Y_test vs Y_train scatter plot


ax[1].set_title('y_test vs y_pred_test')
ax[1].scatter(x=y_test, y=y_pred_test_poly)
ax[1].set_xlabel('y_test')
ax[1].set_ylabel('y_pred_test')
plt.show()

# In[ ]:


errors = {"num_hidden_layers": num_layers,
          "mlp_train_errors": mlp_tr_errors,
          "mlp_val_errors": mlp_val_errors
          }
pd.DataFrame(errors)

# In[ ]:


l_errors = {"poly degree": degrees, "linear_train_errors": linear_tr_errors, "linear_val_errors": linear_val_errors, }
print("training errors and validation errors of PolynomialRegression")
pd.DataFrame(l_errors).style.applymap(
    lambda x: "background-color: yellow" if x == np.min(linear_val_errors) else "background-color: white")

# In[ ]:


errors = {"num_hidden_layers": num_layers,
          "mlp_train_errors": mlp_tr_errors,
          "mlp_val_errors": mlp_val_errors
          }
pd.DataFrame(errors)

# In[ ]:


# create a table to compare training and validation errors
print("training errors, validation errors and test errors of PolynomialRegression")
errors = {"poly degree": degrees,
          "linear_train_errors": linear_tr_errors,
          "linear_val_errors": linear_val_errors,
          "test_error": linear_test_errors
          }
pd.DataFrame({key: pd.Series(value) for key, value in errors.items()})

# In[ ]:


l_errors = {"poly degree": degrees, "linear_train_errors": linear_tr_errors, "linear_val_errors": linear_val_errors,
            "test_error": linear_test_errors}
print("training errors, validation errors and test errors of PolynomialRegression")
pd.DataFrame(l_errors).style.applymap(
    lambda x: "background-color: yellow" if x == np.min(linear_val_errors) else "background-color: white")

# In[ ]:


tr_error = mean_squared_error(y_pred_train_poly, y_train)  # calculate the training error
X_val_poly = poly.fit_transform(X_val)
y_pred_val = lin_regr.predict(X_val_poly)
val_error = mean_squared_error(y_pred_val, y_val)
print("Polynomial degree: ", degree, "Training error: ", tr_error)
print("Polynomial degree: ", degree, "Validation error: ", val_error)

# In[ ]:


# In[ ]:





