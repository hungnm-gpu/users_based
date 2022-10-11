
import pandas as pd
import numpy as np
path = "data_ptit_code.dat"

data = np.loadtxt(path)
# data = np.array([ [1, 1 , 1],
#                 [2, 2, -1],
#                 [2, 2, 1],
#                 [1, 1, -1]])

l = data.shape[0]
train = data[:-int(l/5),:]
test = data[-int(l/5):,:]

def prepare_data(data):
    print(data.shape)
    users = data[:,0]
    items = data[:,1]
    r = data[:,2]
    l = users.shape[0]
    # print(l)
    n_users = int(np.max(data[:,0]))+1
    n_items = int(np.max(data[:, 1]))+1
    # print(n_users)
    dict = {}
    for i in range(l):
        u = int(users[i])
        p = int(items[i])
        if (u,p) in  dict.keys():
            dict[(u,p)] = max(dict[(u,p)] , r[i])
        else:
            dict[(u,p)] = r[i]

    data_matrix = np.array([[0]*n_items]*n_users)
    for (u,p) in dict.keys():
        data_matrix[u][p] = dict[(u,p)]
    print(data_matrix.shape)
    dict.clear()
    return data_matrix


data_train = prepare_data(train)
l_train = data_train.shape[0]
# data_test
__data_test = prepare_data(test)

data_t= []
for i in range(__data_test.shape[0]):
    if i >= l_train:
        continue
    ids = np.where(__data_test[i,:] == 1)[0].astype(np.int32)
    data_t.append([ids])
    # print(ids)
data_test = np.array(data_t)
print(data_test.shape)
    




# def chia_bang(path):
#     df = pd.read_table(path)
#     l = len(df)
#     Dic_person =  {}
#     # print(Dic_person.shape)
#     for i in range(l):
#         u = df.iloc[i,0]
#         p = df.iloc[i,1]
#         tmp = df.iloc[i,2]
#         Dic_person[(u,p)] = tmp
#     return Dic_person


# def main():
#     print("Hello World!")
#     bang = chia_bang(path)
#     print(len(bang))

# if __name__ == "__main__":
#     main()

