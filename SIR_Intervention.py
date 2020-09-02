# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 19:03:50 2020

@author: 91991
"""


# from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

from sklearn import  linear_model, metrics
import pandas as pd
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
import datetime as dt
import matplotlib.dates as mdates
import seaborn as sns
sns.set()

# Value of intervention rho
rho=0.432
# Data upload
df_covid=pd.read_csv('D:/COVID-19_work/states_data_July_27_2020_experiment/Indian States_time_dep_gamma_July_29_2020/India_data_14days/covid19_td_India_14days.csv')
df_covid=df_covid.dropna()
X=df_covid['S.No']
Y=df_covid['Beta']
#Compute the value of beta using linear regression
X=X.values.reshape(-1, 1)
Y=Y.values.reshape(-1,1)
kf = KFold(n_splits=10) # Define the split - into 2 folds 
kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
KFold(n_splits=10, random_state=None, shuffle=False)
X_train=[]
X_test=[]
y_train=[]
y_test=[]
for train_index, test_index in kf.split(X):
    X_train.append(X[train_index])
    X_test.append(X[test_index])
    y_train.append(Y[train_index])
    y_test.append(Y[test_index])
beta11=[]
inter=[]
reg=LinearRegression()
for i in range(0, 10):
    reg.fit(X_train[i],y_train[i])
    inter.append(reg.intercept_)
    beta11.append(reg.coef_)
    
coef=np.mean(beta11)
intercept=np.mean(inter)
beta1=coef*50+intercept


# SIR Model with intervention
def caltr_rate(a,b,c):
    return((a*b)+c)

def deriv(y, t, N1, beta, gamma, rho):
    S, I, R= y
    dSdt = -beta * S * I *(1-rho)/ N1
    dIdt = beta * S * I*(1-rho)/ N1 - gamma * I
    dRdt =gamma * I
    return dSdt, dIdt, dRdt

# intilization of SIR model time dependent
N = df_covid['N']
N=N.values.reshape(-1, 1)
N=N[0]

infacted=df_covid['I (active)']
I0=infacted[50]
De=df_covid['death (cum)']
D0=De[50]
Rec=df_covid['rec(cum)']
R0=Rec[50]

S0 = N - I0 - R0 - D0

# value of gamma and beta
beta2, gamma = beta1, 1./14

# start = dt.datetime(2020,6,5)
# end = dt.datetime(2020,7,26)
start = dt.datetime(2020,7,25)
end = dt.datetime(2020,9,30)
x= [start + dt.timedelta(days=x) for x in range(0, (end-start).days)]
t = np.linspace(0,len(x), len(x))    
# convert arrary of float to int
t=t.astype(int)
y0 = S0, I0, R0 # Initial conditions vector
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta2, gamma, rho))
S, I, R= ret.T
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axisbelow=True)
ax.plot(x, I, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(x, R, 'g', alpha=0.5, lw=2, label='Recovered')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y%b-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=14))
fig.autofmt_xdate(bottom=0.3, rotation=60, ha='right')
ax.set_xlabel('Date')
ax.set_ylabel('COVID-19 cases')
ax.set_ylim(0, (max(R)))
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
plt.title('SIR Model with intervention for India  ')
plt.show()