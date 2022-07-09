Diagnostic of linear regression model
Assumptions of Linear Regression Model
1. Linearity
2. Independence
3. Normality
4. Equal variance

import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

housing = pd.read_csv('data/housing.csv',index_col=0)
housing.head()
# Input the formula (refer to the lecture video 4.3)
formula = formula = 'MEDV~RM'
model = smf.ols(formula=formula, data=housing).fit()

# Here are estimated intercept and slope by least square estimation
# Attribute 'params' returns a list of estimated parameters form model
b0_ols = model.params[0]
b1_ols = model.params[1]

housing['BestResponse'] = b0_ols + b1_ols*housing['RM']

housing['error'] = housing['MEDV'] - housing['BestResponse']
plt.figure(figsize=(15, 8))
plt.title('Residuals vs order')
plt.plot(housing.index, housing['error'], color='purple')
plt.axhline(y=0, color='red')
plt.show()

