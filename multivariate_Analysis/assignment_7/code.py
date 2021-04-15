# Utilizing Numpy and Statsmodels packages

import numpy as np
from statsmodels.multivariate.cancorr import CanCorr

y = [[60, 69, 62],
    [56, 53, 84],
    [80, 69, 76],
    [55, 80, 90],
    [62, 75, 68],
    [74, 64, 70],
    [64, 71, 66],
    [73, 70, 64],
    [68, 67, 75],
    [69, 82, 74],
    [60, 67, 61],
    [70, 74, 78],
    [66, 74, 78],
    [83, 70, 74],
    [68, 66, 90],
    [78, 63, 75],
    [77, 68, 74],
    [66, 77, 68],
    [70, 70, 72],
    [75, 65, 71]]

x = [[97, 69, 98],
    [103, 78, 107],
    [66, 99, 130],
    [80, 85, 114],
    [116, 130, 91],
    [109, 101, 103],
    [77, 102, 130],
    [115, 110, 109],
    [76, 85, 119],
    [72, 133, 127],
    [130, 134, 121],
    [150, 158, 100],
    [150, 131, 142],
    [99, 98, 105],
    [119, 85, 109],
    [164, 98, 138],
    [144, 71, 153],
    [77, 82, 89],
    [114, 93, 122],
    [77, 70, 109]]

y = np.asarray(y)
x = np.asarray(x)

standardized_y = (y - np.mean(y, axis = 0)) / np.std(y, axis = 0)
standardized_x = (x - np.mean(x, axis = 0)) / np.std(x, axis = 0)

S_matrix = np.cov(y.T, x.T)

np.set_printoptions(suppress = True)
print('S matrix = ')
print(np.round(S_matrix, 4))

model = CanCorr(endog = y, exog = x)
model_std = CanCorr(endog = standardized_y, exog = standardized_x)

print('Canonical Correlations between (y_1, y_2, y_3) and (x_1, x_2, x_3): ')
print(np.ndarray.tolist(np.round(model.cancorr, 6)))

print('Squared Canonical Correlations between (y_1, y_2, y_3) and (x_1, x_2, x_3): ')
print(np.ndarray.tolist(np.round(np.square(model.cancorr), 6)))

print('The canonical coefficients for \'endog\' i.e y:')
print(np.round(model.y_cancoef, 6))
print()

print('The canonical coefficients for \'exog\' i.e x:')
print(np.round(model.x_cancoef, 6))

print('Tests of Significance for each Canonical Correlation:')
print()
print(model.corr_test(), end = '')

# ^_^ Thank You