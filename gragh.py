import numpy as np
import pandas as pd
from scipy import sparse


# data = np.array([ [1, 1 , -1],
#                 [3, 2, 1],
#                 [2, 2, 1],
#                 [2, 1, -1]])

df = np.loadtxt("line_2021-07-01_19_09_29_1.dat")

print(df)
data = df


def transfrom(data):
    data_new = data
    n_users = int(np.max(data_new[:, 0]))
    n_items = int(np.max(data_new[:, 1]))
    data_matrix = np.array([[0]*n_items]*n_users)
    data_matrix1 = np.array([[0]*n_items]*n_users)
    data_matrix2 = np.array([[0]*n_items]*n_users)
    users = data_new[:, 0].astype(np.int32)
    items = data_new[:, 1].astype(np.int32)
    ratings = data_new[:, 2]
    for i in range(users.shape[0]):
        data_matrix[users[i]-1][items[i]-1] = ratings[i]
        if(ratings[i] == 1):
            data_matrix1[users[i]-1][items[i]-1] = ratings[i]
        if(ratings[i] == -1):
            data_matrix2[users[i]-1][items[i]-1] = ratings[i]
    return data_matrix, data_matrix1, data_matrix2

def W_L(W):
    _tmp = np.transpose(W)
    TMP = np.matmul(W, _tmp)
    ans =  W
    for i in range(3):
        ans = np.matmul(TMP, ans)
    return ans

W, W1, W2 = transfrom(data)
print(W)
print(W1)
print(W2)
W_L7 = W_L(W)
W_L7_1 = W_L(W1)
W_L7_2 = W_L(W2)
W_L7_3 = W_L7 - W_L7_1 - W_L7_2 

MAX = np.amax(W_L7)
# MAX_1 = np.amax(W_L7_1)
# MAX_2 = np.amax(np.abs(W_L7_2))
# MAX_3 = np.amax(np.abs(W_L7_3))
# MAX_real = max(MAX, MAX_1, MAX_2, MAX_3)
print(MAX)
print(W_L7)
print(W_L7_1)
print(W_L7_2)
print(W_L7_3)
W_L7 = W_L7/MAX
W_L7_1 = W_L7_1 / MAX
W_L7_2 = W_L7_2 / MAX
W_L7_3 = W_L7_3 / MAX
print(W_L7)
print(W_L7_1)
print(W_L7_2)
print(W_L7_3)
def lam_tron(W_L7_1, W_L7_2, W_L7_3):
    x, y = W_L7.shape
    ANS = np.array([[0]*y]*x)
    for i in range(x):
        for j in range(y):
            if(W_L7_1[i][j] > 0.5):
                ANS[i][j] = 1
            if(W_L7_3[i][j] > 0.5):
                ANS[i][j] = 0
            if(W_L7_2[i][j] > 0.5):
                ANS[i][j] = -1
    return ANS
ans = lam_tron(W_L7_1, W_L7_2, W_L7_3)
print(ans)
x, y = ans.shape
for i in range(x):
    print(f'user {i+1}: ')
    for j in range(y):
        if(ans[i][j] == 1):
            print(j+1, end = ' ')
    print()

            



