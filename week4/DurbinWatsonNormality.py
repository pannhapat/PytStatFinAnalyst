import pandas as pd
import statsmodels.formula.api as smf

import scipy.stats as stats
import matplotlib.pyplot as plt

ousing = pd.read_csv('data/housing.csv',index_col=0)
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

z = housing['error'] - housing['error'] .mean()/housing['error'] .std(ddof=1)

stats.probplot(z,dist='norm',plot=plt)
plt.title('Normal Q-Q plot')
plt.show()
