
import pandas as pd
import numpy as np
path = "data_ptit_code.dat"

# data = np.loadtxt(path)
data = np.array([ [1, 1 , 1],
                [2, 2, -1],
                [2, 2, 1],
                [1, 1, -1]])

print(data.shape)
users = data[:,0]
n_users = np.max(data[:,0])
print(n_users)
data_new = []
list_ids = []
for i in range(1,n_users+1):
    ids = np.where(users == i)[0].astype(np.int32)
    rating = data[ids, 2]
    max_rating = np.max(rating)
    if max_rating == 1:
        ids_rating_max = np.where(rating = rating.max)[0].astype(np.int32)
        list_ids

    print(ids)


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


