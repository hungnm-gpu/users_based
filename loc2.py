
import pandas as pd
import numpy as np
path = "line_2021-07-01_19_09_29.dat"
data = np.loadtxt(path)
list_remove = []
def prepare_data(data):
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
        x = int(ids.shape[0])
        if x < 30:
            list_remove.append(i)
            continue
        n = 10
        ids_train = ids[:n]
        ids_test = ids[n:]
        train.extend(data[ids_train])
        test.extend(data[ids_test])

    train = np.array(train)
    test = np.array(test)
    return train, test

train, test = prepare_data(data)

print(test)
print(test.shape)
print(train)
print(train.shape)

users = test[:,0]
items = test[:,1]
ratings = test[:,2]
real_test = []
print(users)
n_users_test = int(np.max(users))+1

for n in range(1, n_users_test):
    ids = np.where(users == n)[0].astype(np.int32)
    # print("ids:", ids)
    tmp = []
    for i in ids:
        if(ratings[i] == 1):
            tmp.append(items[i])
    real_test.append(tmp)

print("test:")
print(real_test)

print("Done_loc2")