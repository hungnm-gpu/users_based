
import pandas as pd
import numpy as np
path = "line_2021-07-01_19_09_29.dat"

df = pd.read_table(path)
data = np.array(df)
print(data.shape)


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


