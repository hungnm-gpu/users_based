import numpy as np 
from collab_filtering import list_recomend
from loc2 import real_test, list_remove

arr1 = list_recomend
arr2 = real_test
# arr1 = np.array([[1,6,3,4,5,2],
#                 [1,6,3,5,2]])
# arr2 = np.array([[1,3,4,10,5],
#                 [1,3,4,5]])


def __MAE(u, pre, real):
    # [1,2,3,4,5,6]
    # [1,3,4,6,5,9]
    pre[u] = np.array(pre[u]).astype(np.int32)
    real[u] = np.array(real[u]).astype(np.int32)
    print("pre_u", pre[u])
    print("real_u", real[u])
    x = np.intersect1d(pre[u], real[u])

    if pre[u].shape[0] == 0:
        return 0
    if real[u].shape[0] == 0:
        return 1
    tmp = min(pre[u].shape[0] , real[u].shape[0])
    print("pre: ")
    print((pre[u].shape[0]))
    print("real: ")
    print((real[u].shape[0]))
    return np.abs(tmp - x.shape[0])/(tmp)

def MAE(n_users):
    ans = 0
    for i in range(n_users):
        # print(__MAE(i,arr1, arr2))
        if i in list_remove:
            continue
        ans+= __MAE(i,arr1, arr2)
    return ans/(n_users-len(list_remove))

print(MAE(len(arr1)))