'''
HW1. Loop over all elements of data array. If the element of data is greater than the value in maximum. Store the current value in maximum.
'''

data = [2,3,5,1,12,21,5,7,9,11,16,0,4]
maximum = 0

for i in data:
  if i > maximum:
    maximum = i

print(maximum)


'''
HW2. Try to make the mean for each column close to zero in the X array.
'''

import numpy as np

X = np.random.randint(10, size=(10, 3))
column_means = np.mean(X , axis=0) # vertical
new_X = X - column_means
print(new_X)

column_sum = 0
for i in range (3):
  for j in range (10):
    column_sum += new_X[j,i]
  print(column_sum)
  column_sum = 0