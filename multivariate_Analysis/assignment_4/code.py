#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Problems on the Analysis of Variance


# ## $Problem\ 1$

# In[2]:


import numpy as np
import scipy as sc
from scipy.stats import f


# In[3]:


data = [[551, 595, 639, 417, 563],
        [457, 580, 615, 449, 631],
        [450, 508, 511, 517, 522],
        [731, 583, 573, 438, 613],
        [499, 633, 648, 415, 656],
        [632, 517, 677, 555, 679]]

data = np.asmatrix(data)
k = data.shape[1]
n = data.shape[0]

print('k = ' + str(k))
print('n = ' + str(n))


# In[4]:


total_mean = np.mean(data)
row_means = np.asarray(np.ravel(np.mean(data, axis = 0)))

print('Total data mean = ' + str(total_mean))
print('Row means = ' + str(np.round(row_means, 4)))


# In[5]:


numerator = (n / (k - 1)) * np.matmul((row_means - total_mean), 
                                      (row_means - total_mean))
denominator = 0

for j in range(n):
  for i in range(k):
    denominator += np.power((data[j, i] - row_means[i]), 2)

denominator /= (k * (n - 1))

F_value = numerator / denominator
alpha = 0.05
F_critical = f.isf(alpha, k - 1, k * (n - 1))

print('Computed F Value = ' + str(F_value))
print('Critical F Value = ' + str(F_critical))
print('Computed F value > Critical F value = ' + str(F_value > F_critical))


# $ We\ reject\ H_0.$

# ## $Problem\ 2$

# In[6]:


set_1 = np.asmatrix([[1.11,	2.569,	3.58,	0.760],
                    [1.19,	2.928,	3.75,	0.821],
                    [1.09,	2.865,	3.93,	0.928],
                    [1.25,	3.844,	3.94,	1.009],
                    [1.11,	3.027,	3.60,	0.766],
                    [1.08,	2.336,	3.51,	0.726],
                    [1.11,	3.211,	3.98,	1.209],
                    [1.16,	3.037,	3.62,	0.750]])

set_2 = np.asmatrix([[1.05,	2.074,	4.09,	1.036],
                    [1.17,	2.885,	4.06,	1.094],
                    [1.11,	3.378,	4.87,	1.635],
                    [1.25,	3.906,	4.98,	1.517],
                    [1.17,	2.782,	4.38,	1.197],
                    [1.15,	3.018,	4.65,	1.244],
                    [1.17,	3.383,	4.69,	1.495],
                    [1.19,	3.447,	4.40,	1.026]])

set_3 = np.asmatrix([[1.07,	2.505,	3.76,	0.912],
                    [0.99,	2.315,	4.44,	1.398],
                    [1.06,	2.667,	4.38,	1.197],
                    [1.02,	2.390,	4.67,	1.613],
                    [1.15,	3.021,	4.48,	1.476],
                    [1.20,	3.085,	4.78,	1.571],
                    [1.20,	3.308,	4.57,	1.506],
                    [1.17,	3.231,	4.56,	1.458]])

    
set_4 = np.asmatrix([[1.22,	2.838,	3.89,	0.944],
                    [1.03,	2.351,	4.05,	1.241],
                    [1.14,	3.001,	4.05,	1.023],
                    [1.01,	2.439,	3.92,	1.067],
                    [0.99,	2.199,	3.27,	0.693],
                    [1.11,	3.318,	3.95,	1.085],
                    [1.20,	3.601,	4.27,	1.242],
                    [1.08,	3.291,	3.85,	1.017]])
    
set_5 = np.asmatrix([[0.91,	1.532,	4.04,	1.084],
                    [1.15,	2.552,	4.16,	1.151],
                    [1.14,	3.083,	4.79,	1.381],
                    [1.05,	2.330,	4.42,	1.242],
                    [0.99,	2.079,	3.47,	0.673],
                    [1.22,	3.366,	4.41,	1.137],
                    [1.05,	2.416,	4.64,	1.455],
                    [1.13,	3.100,	4.57,	1.325]])
    
set_6 = np.asmatrix([[1.11,	2.813,	3.76,	0.800],
                    [0.75,	0.840,	3.14,	0.606],
                    [1.05,	2.199,	3.75,	0.790],
                    [1.02,	2.132,	3.99,	0.853],
                    [1.05,	1.949,	3.34,	0.610],
                    [1.07,	2.251,	3.21,	0.562],
                    [1.13,	3.064,	3.63,	0.707],
                    [1.11,	2.469,	3.95,	0.952]])

data = [set_1, set_2, set_3, set_4, set_5, set_6]
total = np.vstack([set_1, set_2, set_3, set_4, set_5, set_6])

p = np.shape(set_1)[1]
n = np.shape(set_1)[0]
k = len(data)
N = n * k

print('p = ' + str(p))
print('n = ' + str(n))
print('k = ' + str(k))
print('N = ' + str(N))


# ### $(a)$

# In[7]:


means = []
print('Unit means: ')
print()

for index in range(len(data)):
  means.append(np.mean(data[index], axis = 0))
  print('mean[i = ' + str(index + 1) + '] = ' + str(np.ravel(means[-1])))

mean_all = np.mean(total, axis = 0)

print()
print('Total mean: ' + str(np.ravel(np.round(mean_all, 4))))


# In[8]:


H = 0
for i in range(k):
  H += np.matmul((means[i] - mean_all).T, (means[i] - mean_all))

H = n * H
print('H = ')
print()
print(np.round(H, 4))


# In[9]:


E = 0
for i in range(k):
  for j in range(n):
    unit = data[i][j] - means[i]
    E += np.matmul(unit.T, unit)

print('E = ')
print()
print(np.round(E, 4))


# In[10]:


ratio = np.linalg.det(E) / np.linalg.det(E + H)
nu_H = k - 1
nu_E = k * (n - 1)

print('p = ' + str(p))
print('nu_H = ' + str(nu_H))
print('nu_E = ' + str(nu_E))


# In[11]:


w = nu_E + nu_H - (p + nu_H + 1) / 2
t = np.sqrt((p * p * nu_H * nu_H - 4) / (p * p + nu_H * nu_H - 5))
df_1 = float(p * nu_H)
df_2 = w * t - (p * nu_H - 2) / 2

print('w = ' + str(w))
print('t = ' + str(t))
print('df_1 = ' + str(df_1))
print('df_2 = ' + str(df_2))


# In[12]:


step = np.power(ratio, (1 / t))
F_calculated = ((1 - step) / step) * (df_2 / df_1)
alpha = 0.05
F_compare = f.isf(alpha, df_1, df_2)


# In[13]:


print('F_calculated = ' + str(F_calculated))
print('F_compare = ' + str(F_compare))
print('F_calculated > F_compare = ' + str(F_calculated > F_compare))


# $We\ reject\ H_0.$

# ### $(b)$

# In[14]:


def T_square_value(alpha, p, v):
  # 'v' being the number of degrees of freedom 
  # 'p' - variable setting

  return f.isf(alpha, p, v - p + 1) / (v - p + 1) * (v * p)


# In[15]:


C_1 = [2, -1, -1, -1, -1, 2]
delta_1 = 0

for index in range(len(C_1)):
  delta_1 += C_1[index] * means[index]

sum_C1_square = np.matmul(np.asmatrix(C_1), np.asmatrix(C_1).T)

T_1_squared = (n / sum_C1_square) * np.matmul(
    np.matmul(delta_1, np.linalg.inv(E / nu_E)), delta_1.T)

T_1_squared = np.ravel(T_1_squared)[0]

alpha = 0.05 / 2
# Required Adjustment

T_1_square_comp = T_square_value(alpha, p, nu_E)

print('delta_1 = ' + str(np.round(np.ravel(delta_1), 4)))
print('T_1_squared = ' + str(T_1_squared))
print('T_1_square_comp = ' + str(T_1_square_comp))
print('Calculated T square > T square value for comparison: '
      + str(T_1_squared > T_1_square_comp))


# $We\ reject\ H_0\ for\ C_1.$

# In[16]:


C_2 = [1, 0, 0, 0, 0, -1]
delta_2 = 0

for index in range(len(C_2)):
  delta_2 += C_2[index] * means[index]

sum_C2_square = np.matmul(np.asmatrix(C_2), np.asmatrix(C_2).T) 

T_2_squared = (n / sum_C2_square) * np.matmul(
    np.matmul(delta_2, np.linalg.inv(E / nu_E)), delta_2.T)

T_2_squared = np.ravel(T_2_squared)[0]

alpha = 0.05 / 2
# Required Adjustment

T_2_square_comp = T_square_value(alpha, p, nu_E)

print('delta_2 = ' + str(np.round(np.ravel(delta_2), 4)))
print('T_2_squared = ' + str(T_2_squared))
print('T_2_square_comp = ' + str(T_2_square_comp))
print('Calculated T square > T square value for comparison: '
      + str(T_2_squared > T_2_square_comp))


# $We\ reject\ H_0\ for\ C_2.$

# ### $(c)$

# In[17]:


# Testing Individual Variables

alpha = 0.05
n = 8
k = 6

for variable in range(4):
  print('Testing for y' + str(variable + 1) + ":")
  print()
  data = np.vstack([np.ravel(set_1[:, variable].T),
                  np.ravel(set_2[:, variable].T),
                  np.ravel(set_3[:, variable].T),
                  np.ravel(set_4[:, variable].T),
                  np.ravel(set_5[:, variable].T),
                  np.ravel(set_6[:, variable].T)])

  data = data.T

  assert n == data.shape[0]
  assert k == data.shape[1]

  total_mean = np.mean(data)
  row_means = np.asarray(np.ravel(np.mean(data, axis = 0)))

  print('Total data mean = ' + str(total_mean))
  print('Row means = ' + str(np.round(row_means, 4)))

  numerator = (n / (k - 1)) * np.matmul(
      (row_means - total_mean), (row_means - total_mean))
  
  denominator = 0

  for j in range(n):
    for i in range(k):
      denominator += np.power((data[j, i] - row_means[i]), 2)

  denominator /= (k * (n - 1))

  F_value = numerator / denominator
  F_critical = f.isf(alpha, k - 1, k * (n - 1))

  print('Computed F Value = ' + str(F_value))
  print('Critical F Value = ' + str(F_critical))
  print('Computed F value > Critical F value = ' + str(F_value > F_critical))

  print()
  if F_value > F_critical:
    print('We reject H_0 for y' + str(variable + 1) + '.')
  else:
    print('We accept H_0 for y' + str(variable + 1) + '.')
  print()


# $Decisions:$
# 
# - $Accept\ H_0\ for\ y1.$
# 
# - $Reject\ H_0\ for\ y2.$
# 
# - $Reject\ H_0\ for\ y3.$
# 
# - $Reject\ H_0\ for\ y4.$

# ## $Problem\ 3$

# In[18]:


import json

data_a = [[0.71, 2.20, 2.25],
          [1.66, 2.93, 3.93],
          [2.01, 3.08, 5.08],
          [2.16, 3.49, 5.82],
          [2.42, 4.11, 5.84],
          [2.42, 4.95, 6.89],
          [2.56, 5.16, 8.50],
          [2.60, 5.54, 8.56],
          [3.31, 5.68, 9.44],
          [3.64, 6.25, 10.52],
          [3.74, 7.25, 13.46],
          [3.74, 7.90, 13.57],
          [4.39, 8.85, 14.76],
          [4.50, 11.96, 16.41],
          [5.07, 15.54, 16.96],
          [5.26, 15.89, 17.56],
          [8.15, 18.3, 22.82],
          [8.24, 18.59, 29.13]]

data_b = [[2.20, 4.04, 2.71],
          [2.69, 4.16, 5.43],
          [3.54, 4.42, 6.38],
          [3.75, 4.93, 6.38],
          [3.83, 5.49, 8.32],
          [4.08, 5.77, 9.04],
          [4.27, 5.86, 9.56],
          [4.53, 6.28, 10.01],
          [5.32, 6.97, 10.08],
          [6.18, 7.06, 10.62],
          [6.22, 7.78, 13.80],
          [6.33, 9.23, 15.99],
          [6.97, 9.34, 17.90],
          [6.97, 9.91, 18.25],
          [7.52, 13.46, 19.32],
          [8.36, 18.4, 19.87],
          [11.65, 23.89, 21.60],
          [12.45, 26.39, 22.25]]

data_a = np.asmatrix(data_a)
data_b = np.asmatrix(data_b)

# Using a natural logarithm transformation
data_a = np.log(data_a)
data_b = np.log(data_b)

all_data = [data_a, data_b]


# $2\ types\ of\ iron\ used.$
# $3\ molarities\ for\ each\ type.$
# $18\ observations\ per\ cell.$

# In[19]:


mean_total = np.mean(np.vstack(all_data))

# i, j, k:
# i = 1 or 2 (types of Iron)
# j = 1, 2, or 3 (types of molarities)
# k = 1, 2, ... , 18 (observation index)

a = 2
b = 3
n = 18


# In[20]:


mean_1__ = np.mean(data_a)
mean_2__ = np.mean(data_b)

mean_i__ = np.asarray([mean_1__, mean_2__])

print('Mean of A Chunks: ')
print(mean_i__)
print()

mean__j_ = np.ravel(np.mean(np.vstack([data_a, data_b]), axis = 0))

print('Mean of B Chunks: ')
print(mean__j_)


# In[21]:


step = mean_i__ - mean_total
SSA = n * b * np.matmul(step, step.T)

print('SSA = ' + str(SSA))


# In[22]:


step = mean__j_ - mean_total
SSB = n * a * np.matmul(step, step.T)
print('SSB = ' + str(SSB))

mean_1j_ = np.mean(data_a, axis = 0)
mean_2j_ = np.mean(data_b, axis = 0)
box_means = np.ndarray.tolist(mean_1j_) + np.ndarray.tolist(mean_2j_)


# In[23]:


SSAB = 0
for i in range(a):
  for j in range(b):
    unit = box_means[i][j] - mean_i__[i] - mean__j_[j] + mean_total
    SSAB += unit * unit

SSAB = n * SSAB
print('SSAB = ' + str(SSAB))


# In[24]:


SSE = 0
for i in range(a):
  for j in range(b):
    for k in range(n):
      unit = all_data[i][k, j] - box_means[i][j]
      SSE += unit * unit

print('SSE = ' + str(SSE))

SST = 0
for i in range(a):
  for j in range(b):
    for k in range(n):
      unit = all_data[i][k, j] - mean_total
      SST += unit * unit

print('SST = ' + str(SST))

df = [a - 1, b - 1, (a - 1) * (b - 1), a * b * (n - 1), a * b * n - 1]

MSA = SSA / df[0]
MSB = SSB / df[1]
MSAB = SSAB / df[2]
MSE = SSE / df[3]

F_A = MSA / MSE
F_B = MSB / MSE
F_AB = MSAB / MSE


# In[25]:


print('SSA + SSB + SSAB + SSE = ' + str(SSA + SSB + SSAB + SSE))
print('SST = ' + str(SST))


# $\therefore\ SSA\ +\ SSB\ +\ SSAB\ +\ SSE\ =\ SST$

# In[26]:


alpha = 0.01
p_comp_A = 1 - f.cdf(F_A, df[0], df[3])
p_comp_B = 1 - f.cdf(F_B, df[1], df[3])
p_comp_AB = 1 - f.cdf(F_AB, df[2], df[3])


# In[27]:


python_print = False

print('ANOVA Table:')
print()

if python_print:
    print('Source\tSS\tDF\tMS\tF\tp-value')

    print('Iron\t' + str(np.round(SSA, 4)) + '\t' + str(df[0]) + '\t' + 
          str(np.round(MSA, 4)) + '\t' + str(np.round(F_A, 4)) +
          '\t' + str(np.round(p_comp_A, 4)))

    print('Mol\t' + str(np.round(SSB, 4)) + '\t' + str(df[1]) + '\t' + 
          str(np.round(MSB, 4)) + '\t' + str(np.round(F_B, 4)) +
          '\t' + str(np.round(p_comp_B, 4)))

    print('Int.\t' + str(np.round(SSAB, 4)) + '\t' + str(df[2]) + 
          '\t' + str(np.round(MSAB, 4)) + '\t' + str(np.round(F_AB)) +
          '\t' + str(np.round(p_comp_AB, 4)))

    print('Error\t' + str(np.round(SSE, 4)) + '\t' + str(df[3]) + 
          '\t' + str(np.round(MSE, 4)))

    print('Total\t' + str(np.round(SST, 4)) + '\t' + str(df[4]))

else:
    # For pretty-printing the above table
    import pandas as pd

    data = {'SS': ['2.0738', '15.5884', '0.8103', '35.2959', '53.7684'],
            'DF': ['1', '2', '2', '102', '107'],
            'MS': ['2.0738', '7.7942', '0.4051', '0.346', '-'],
            'F': ['5.9931', '22.5241', '1.0', '-', '-'],
            'p-value': ['0.0161', '0.0', '0.3143', '-', '-']
            }

    dataframe = pd.DataFrame(data, 
                columns = ['SS', 'DF', 'MS', 'F', 'p-value'],
                index = ['Iron', 'Molarity', 'Interact', 'Error', 'Total'])
    print(dataframe)


# ## $Problem\ 4$

# In[28]:


A1_1 = [7.80, 7.10, 7.89, 7.82, 9.00, 8.43, 7.65, 7.70,
        7.28, 8.96, 7.75, 7.80, 7.60, 7.00, 7.82, 7.80]
A1_2 = [90.4, 88.9, 85.9, 88.8, 82.50, 92.40, 82.40, 87.40,
        79.60, 95.10, 90.20, 88.00, 94.1, 86.6, 85.9, 88.8]

A2_1 = [7.12, 7.06, 7.45, 7.45, 8.19, 8.25, 7.45, 7.45,
        7.15, 7.15, 7.70, 7.45, 7.06, 7.04, 7.52, 7.70]
A2_2 = [85.1, 89.0, 75.9, 77.9, 66.0, 74.5, 83.1, 86.4,
        81.2, 72.0, 79.9, 71.9, 81.2, 79.9, 86.4, 76.4]

A1_1 = np.asarray(A1_1)
A1_2 = np.asarray(A1_2)
A2_1 = np.asarray(A2_1)
A2_2 = np.asarray(A2_2)


# In[29]:


A1 = np.asmatrix([A1_1, A1_2]).T
A2 = np.asmatrix([A2_1, A2_2]).T


# In[30]:


p = 2
a = 2
b = 4
n = 4

all_data = [A1, A2]
mean_total = np.ravel(np.mean(np.vstack(all_data), axis = 0))

print('Mean of all observations: ' + str(mean_total))
print()

mean_1__ = np.mean(A1, axis = 0)
mean_2__ = np.mean(A2, axis = 0)

mean_i__ = np.vstack([mean_1__, mean_2__])
print('Mean of A Chunks: ')
print(mean_i__)
print()

mean__j_ = []

print('Mean of B Chunks:')
print()
for index in range(b):
  blob = np.vstack([A1[4 * index : 4 * index + 4], A2[4 * index : 4 * index + 4]]) 
  mean__j_.append(np.ravel(np.mean(blob, axis = 0)))
  print('mean[j = ' + str(index + 1) + '] = ' + str(mean__j_[-1]))


# In[31]:


mean_ij_ = []
print('Means of i, j chunks:')
print()

for i in range(a):
  mean_ij_.append([])
  for j in range(b):
    blob = all_data[i][4 * j : 4 * j + 4]
    mean_ij_[i].append(np.ravel(np.mean(blob, axis = 0)))
    print('mean[i = ' + str(i + 1) + '][j = ' + str(j + 1) + '] = '
          + str(mean_ij_[i][j]))


# In[32]:


H_A = np.zeros([p, p])

for i in range(a):
  step = mean_i__[i, :] - mean_total
  H_A += np.matmul(step.T, step)
H_A = n * b * H_A

print('H_A = ')
print()
print(H_A)


# In[33]:


H_B = np.zeros([p, p])

for j in range(b):
  unit = np.asmatrix(mean__j_[j] - mean_total)
  H_B += np.matmul(unit.T, unit)

H_B = n * a * H_B
print('H_B = ')
print()
print(H_B)


# In[34]:


H_AB = np.zeros([p, p])

for i in range(a):
  for j in range(b):
    unit = mean_ij_[i][j] - mean_i__[i] - mean__j_[j] + mean_total
    unit = np.asmatrix(unit)
    H_AB += np.matmul(unit.T, unit)

H_AB = n * H_AB
print('H_AB = ')
print()
print(H_AB)


# In[35]:


E = np.zeros([p, p])

for i in range(a):
  for j in range(b):
    for k in range(n):
      unit = np.asmatrix(all_data[i][4 * j + k] - mean_ij_[i][j])
      E += np.matmul(unit.T, unit)

print('E = ')
print()
print(E)


# In[36]:


T = np.zeros([p, p])

for i in range(a):
  for j in range(b):
    for k in range(n):
      unit = np.asmatrix(all_data[i][4 * j + k] - mean_total)
      T += np.matmul(unit.T, unit)

print('T = ')
print()
print(T)


# In[37]:


# Tests 
nu_H = a - 1
nu_E = a * b * (n - 1) 
multiplier = (nu_E + nu_H - p) / p
alpha = 0.05

# Test for interaction
cap_AB = np.linalg.det(E) / np.linalg.det(E + H_AB)
F_AB = ((1 - cap_AB) / cap_AB) * multiplier
F_comp_AB = f.isf(alpha, p, nu_E + nu_H - p)

print('cap_AB = ' + str(cap_AB))
print('Calculated F_AB = ' + str(F_AB))
print('F_AB for comparison = ' + str(F_comp_AB))
print('Calculated F_AB > F_AB for comparison: ' + str(F_AB > F_comp_AB))

# Test for A's effect
cap_A = np.linalg.det(E) / np.linalg.det(E + H_A)
F_A = ((1 - cap_A) / cap_A) * multiplier
F_comp_A = f.isf(alpha, p, nu_E + nu_H - p)

print()
print('cap_A = ' + str(cap_A))
print('Calculated F_A = ' + str(F_A))
print('F_A for comparison = ' + str(F_comp_A))
print('Calculated F_A > F_A for comparison: ' + str(F_A > F_comp_A))

# Test for B's effect
cap_B = np.linalg.det(E) / np.linalg.det(E + H_B)
F_B = ((1 - cap_B) / cap_B) * multiplier
F_comp_B = f.isf(alpha, p, nu_E + nu_H - p)

print()
print('cap_B = ' + str(cap_B))
print('Calculated F_B = ' + str(F_B))
print('F_B for comparison = ' + str(F_comp_B))
print('Calculated F_B > F_B for comparison: ' + str(F_B > F_comp_B))


# $Conclusions:$
# 
# - $No\ Factor\ A\ \ effects.$
# 
# - $No\ Factor\ B\ \ effects.$
# 
# - $Interaction\ effects\ exist.$

# In[38]:


# ^_^ Thank You

