import numpy as np 

arr1 = np.array([[1,6,3,4,5,2],
                [1,6,3,4,5,2]])
arr2 = np.array([[1,3,4,10,5],
                [1,3,4,6,5]])

arr1 = np.append(arr1, [[1,2,3]], axis=0)
print(arr1)
# def __MAE(u, pre, real):
#     # [1,2,3,4,5,6]
#     # [1,3,4,6,5,9]
#     x = np.intersect1d(pre[u], real[u])
#     if real[u].shape[0] == 0:
#         if pre[u].shape[0] == 0:
#             return 0
#         return 1
#     return np.abs(real[u].shape[0] - x.shape[0])/(real[u].shape[0])

# def MAE(n_users):
#     ans = 0
#     for i in range(n_users):
#         print(__MAE(i,arr1, arr2))
#         ans+= __MAE(i,arr1, arr2)
#     return ans/n_users

# print(MAE(arr1.shape[0]))