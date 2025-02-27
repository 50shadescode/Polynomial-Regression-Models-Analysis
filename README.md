Polynomial Regression Models Analysis

1. Data Preprocessing

Step 1: Import Required Libraries

Before working on our model, let's import the essential libraries.

import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  

Step 2: Load the Dataset

We will use the 50_Startups.csv dataset, which contains information about R&D Spend, Administration, Marketing Spend, and Profit.

df = pd.read_csv("50_Startups.csv")

Step 3: Explore the Data

Understanding the dataset before applying any model is crucial.

# Display dataset information
df_info = df.info()
df_description = df.describe()

Note: The dataset does not contain any missing values.

Step 4: Handle Categorical Data

Since "State" is a categorical variable, we need to convert it into numerical format using one-hot encoding.

# Encode Categorical Variables
df = pd.get_dummies(df, columns=['State'], drop_first=True)

2. Simple Linear Regression

Objective: Predict Profit using a single independent variable (R&D Spend).

Step 1: Select Features and Target Variable

X = df[['R&D Spend']]  
y = df['Profit']

Step 2: Split Data into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Why? Training data is used to build the model, while testing data evaluates its performance.

Step 3: Train the Model

model = LinearRegression()
model.fit(X_train, y_train)

Step 4: Make Predictions

y_pred = model.predict(X_test)

Step 5: Evaluate the Model

mae = mean_absolute_error(y_test, y_pred)  
mse = mean_squared_error(y_test, y_pred)  
rmse = np.sqrt(mse)  
r2 = r2_score(y_test, y_pred)  

print(f"MAE: {mae:.2f}")  
print(f"MSE: {mse:.2f}")  
print(f"RMSE: {rmse:.2f}")  
print(f"R2 Score: {r2:.2f}")  

Output:MAE: 6077.36MSE: 59510962.81RMSE: 7714.33R2 Score: 0.93

Step 6: Visualize the Regression Line

plt.scatter(X_test, y_test, color='blue', label='Actual Data')  
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')  
plt.xlabel("R&D Spend")  
plt.ylabel("Profit")  
plt.title("Simple Linear Regression")  
plt.legend()  
plt.show()

Final Interpretation:

The high R² score (0.93) suggests that R&D Spend explains 93% of the variance in Profit, meaning it's a strong predictor.

The relatively low RMSE (7714.33) indicates that the average prediction error is around $7,714.

The regression line follows the trend of actual data points, confirming a strong linear relationship between R&D Spend and Profit.

Next Steps:

Check Residual Patterns.

Try Multiple Regression.

Consider Polynomial Regression.

3. Multiple Linear Regression

Objective: Predict Profit using multiple independent variables.

Step 1: Select Features and Target Variable

X_multi = df[['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida']]
y_multi = df['Profit']

Step 2: Train and Evaluate the Model

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)

y_pred_multi = model_multi.predict(X_test_m)

mae_multi = mean_absolute_error(y_test_m, y_pred_multi)
mse_multi = mean_squared_error(y_test_m, y_pred_multi)
rmse_multi = np.sqrt(mse_multi)
r2_multi = r2_score(y_test_m, y_pred_multi)

Output:MAE: 6961.49MSE: 82010370.81RMSE: 9055.96R2 Score: 0.90

Interpretation:

The model explains 90% of the variance in Profit.

RMSE (9055.96) & MAE (6961.49) indicate the average error in predictions.

While the model performs well, there may be outliers affecting predictions.

4. Polynomial Regression

Objective: Model a non-linear relationship between R&D Spend and Profit.

Step 1: Transform Data for Polynomial Regression

degree = 2
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(df[['R&D Spend']])

Step 2: Train and Evaluate the Model

model_poly = LinearRegression()
model_poly.fit(X_poly, y)
y_pred_poly = model_poly.predict(X_poly)

Step 3: Visualize the Curve

plt.scatter(df['R&D Spend'], y, color='blue', label='Actual Data')  
plt.scatter(df['R&D Spend'], y_pred_poly, color='red', label='Polynomial Predictions')  
plt.xlabel("R&D Spend")  
plt.ylabel("Profit")  
plt.title("Polynomial Regression (Degree = 2)")  
plt.legend()  
plt.show()

Interpretation:

Polynomial Regression (degree = 2) captures the non-linear relationship between R&D Spend and Profit. The model follows the data trend more closely than Simple Linear Regression.

Conclusion:
This analysis covers data preprocessing, simple linear regression, multiple linear regression, and polynomial regression to predict company profits. The results demonstrate how different regression techniques can be used to analyze relationships between features and target variables.
