# Script que carga un archivo de datos csv
# y hace la regresión lineal con dos variables independientes
# esos resultados fueron cruzados con el resultado en Octave
# archivo Octave\practice\TwoVariableLinearFit.m
import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv('2-variable-dataset.csv', delimiter=",")
X = df.values[:, :-1]
Y = df.values[:, -1]
fit = LinearRegression().fit(X, Y)
print("z = {0} + {1}x + {2}y".format(fit.intercept_, fit.coef_[0], fit.coef_[1]))
print(fit.coef_[0]*X[:,0]+fit.coef_[1]*X[:,1]+fit.intercept_)
