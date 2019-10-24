from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def linear_regression(data, power, models_to_plot):
    #initialize predictors:
    predictors=['x']
    if power>=2:
        predictors.extend(['x_%d'%i for i in range(2,power+1)])

    #Fit the model
    linreg = LinearRegression(normalize=True)
    linreg.fit(data[predictors],data['y'])
    y_pred = linreg.predict(data[predictors])

    #Check if a plot is to be made for the entered power
    if power in models_to_plot:
        plt.subplot(models_to_plot[power])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for power: %d'%power)

    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([linreg.intercept_])
    ret.extend(linreg.coef_)
    return ret


def ridge_regression(data, predictors, alpha, models_to_plot={}):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(data[predictors],data['y'])
    y_pred = ridgereg.predict(data[predictors])

    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for alpha: %.3g'%alpha)

    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret

def lasso_regression(data, predictors, alpha, models_to_plot={}):
    #Fit the model
    lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e3)
    lassoreg.fit(data[predictors],data['y'])
    y_pred = lassoreg.predict(data[predictors])

    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        for i in range(N):
            plt.plot(data['x'][indicators[i] == 1],data['y'][indicators[i] == 1],'.')
            plt.plot(data['x'][indicators[i] == 1],y_pred[indicators[i] == 1])
        plt.title('Plot for alpha: %.3g'%alpha)

    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret

time, mag, mag_err, curve_num = np.loadtxt('test7.csv')[:,2000:2500]
## Generate 3 different curves with random intercept and noise
x = time - 7511.
y = mag
N = len(np.unique(curve_num))
indicators = np.zeros((N, len(x)))
for dum in range(N):
    indicators[dum][np.where(curve_num == dum)[0]] = 1


max_pwr = 12

data = pd.DataFrame(np.column_stack([x,y]),columns=['x','y'])
plt.plot(data['x'],data['y'],'.')

for i in range(2,max_pwr):  #power of 1 is already there
    colname = 'x_%d'%i      #new var will be x_power
    data[colname] = data['x']**i
for i in range(max_pwr, max_pwr + N):
    colname = 'intrp_%d'%(i-max_pwr)
    data[colname] = indicators[i - max_pwr]
print data.head()

predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,max_pwr)])
predictors.extend(['intrp_%d'%i for i in range(N)])

#Define the alpha values to test
alpha_lasso = [0, 1e-30, 1e-25, 1e-20, 1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3]

#Initialize the dataframe to store coefficients
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,max_pwr)] + ['interp_%d'%i for i in range(max_pwr,max_pwr+N)]
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

#Define the models to plot
models_to_plot = {0:231, 1e-25:232,1e-15:233, 1e-8:234, 1e-5:235, 1e-3:236}

#Iterate over the 10 alpha values:
for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(data, predictors, alpha_lasso[i], models_to_plot)


coef_matrix_lasso.apply(lambda x: sum(x.values==0),axis=1)
