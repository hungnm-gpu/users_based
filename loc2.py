
import pandas as pd
import numpy as np
path = "line_2021-07-01_19_09_29.dat"
data = np.loadtxt(path)

def prepare_data(data, n):
    train = []
    test = []
    users = data[:,0]
    items = data[:,1]
    r = data[:,2]
    l = users.shape[0]
    # print(l)
    n_users = int(np.max(data[:,0]))+1
    n_items = int(np.max(data[:, 1]))+1
    # print(n_users)
    for i in range(1, n_users):
        ids = np.where(users == i)[0].astype(np.int32)
        if (ids.shape[0] < n):
            continue
        ids_train = ids[:n]
        ids_test = ids[n:]
        train.extend(data[ids_train])
        test.extend(data[ids_test])

    train = np.array(train)
    test = np.array(test)
    return train, test

train, test = prepare_data(data, 20)
print("Done_loc2")
print(test)
print(test.shape)
print(train)
print(train.shape)