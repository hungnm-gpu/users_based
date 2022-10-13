from operator import mod
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sympy import I


class CF(object):
    """
    class Collaborative Filtering, hệ thống đề xuất dựa trên sự tương đồng
    giữa các users với nhau, giữa các items với nhau
    """

    def __init__(self, data_matrix, k=2, dist_func=cosine_similarity, uuCF=0):
        """
        Khởi tạo CF với các tham số đầu vào:
            data_matrix: ma trận Utility, gồm 3 cột, mỗi cột gồm 3 số liệu: user_id, item_id, rating.
            k: số lượng láng giềng lựa chọn để dự đoán rating.
            uuCF: Nếu sử dụng uuCF thì uuCF = 1 , ngược lại uuCF = 0. Tham số nhận giá trị mặc định là 1.
            dist_f: Hàm khoảng cách, ở đây sử dụng hàm cosine_similarity của klearn.
            limit: Số lượng items gợi ý cho mỗi user. Mặc định bằng 10.
        """
        self.uuCF = uuCF  # user-user (1) or item-item (0) CF
        self.Y_data = data_matrix if uuCF else data_matrix[:, [1, 0, 2]]
        self.Y_data = np.array(self.Y_data, dtype=np.float64)
        self.k = k
        self.dist_func = dist_func
        self.Ybar_data = None
        # số lượng user và item, +1 vì mảng bắt đầu từ 0
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1
        print(self.n_users, self.n_items)

    def add(self, new_data):
        """
        Cập nhật Y_data khi có lượt rating mới.
        """
        self.Y_data = np.concatenate((self.Y_data, new_data), axis=0)

    def normalize_matrix(self):
        """
        Tính similarity giữa các items bằng cách tính trung bình cộng ratings giữa các items.
        Sau đó thực hiện chuẩn hóa bằng cách trừ các ratings đã biết của item cho trung bình cộng
        ratings tương ứng của item đó, đồng thời thay các ratings chưa biết bằng 0.
        """
        users = self.Y_data[:, 0]  # trả về cột đầu tiên trong data
        print(users.shape)

        self.Ybar_data = self.Y_data.copy()  # copy ma trận đầu vào
        self.mu = np.zeros((self.n_users,))  # khởi tạo mảng 0 có đọ dài = n_users
        check = 0
        for n in range(1, self.n_users):
            # if n in list_recomend:
            #     continue
            ids = np.where(users == n)[0].astype(np.int32)  # trả về các vị trí của user = n
            # ids 3
            item_ids = self.Y_data[ids, 1]
            ratings = self.Y_data[ids, 2]
            ratings = np.array(ratings, dtype=np.float64)
            # take mean
            m = np.mean(ratings)
            if np.isnan(m):
                m = 0  # để tránh mảng trống và nan value
            self.mu[n] = m
            # chuẩn hóa
            self.Ybar_data[ids, 2] = ratings - self.mu[n]
            # if check == 0:
            #     print(tmp)
            #     print(self.Ybar_data[ids, 2])
            #     check = 1
        # print(type(self.Ybar_data[ids, 2][1]))
        # print(type(self.mu[n]))
        # print(self.Ybar_data[ids, 2])
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
                                       (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()
        print(self.Ybar[1, 1])

    def similarity(self):
        """
        Tính độ tương đồng giữa các user và các item
        """
        eps = 1e-6
        # tính toán độ tương đồng của cặp user theo item và rating
        print("self.Ybar.T: ", self.Ybar.T)
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)
        print(self.S[:10,:10])
        print(self.S.shape)

    def fit(self):
        self.normalize_matrix()
        self.similarity()

    def __pred(self, u, i, normalized=1):
        """
        Dự đoán ra ratings của các users với mỗi items.
        u là item
        i là user
        """
        # tìm tất cả item đã được rate bởu user i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        # print(users_rated_i)
        sim = self.S[u, users_rated_i]
        a = np.argsort(sim)[-self.k:]
        nearest_s = sim[a]
        r = self.Ybar[i, users_rated_i[a]]
        # print(f"neatest {u} vs {users_rated_i[a]}")
        # print(nearest_s)
        # print(r)
        if normalized:
            # cộng với 1e-8, để tránh chia cho 0
            return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8)
        return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8) + self.mu[u]

    def pred(self, u, i, normalized=1):
        """
        Xét xem phương pháp cần áp dùng là uuCF hay iiCF
        """
        if self.uuCF: return self.__pred(u, i, normalized)
        return self.__pred(i, u, normalized)

    def print_list_item(self):
        for i in range(self.n_items):
            print(i)

    def recommend(self, u, limit=10):
        ids = np.where(self.Y_data[:, 1] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()
        recommended_items = np.array([])
        rating_for_sort = np.array([])
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                # print(f'ranking item {i}: {rating}' )
                if rating > 0:
                    rating_for_sort = np.append(rating_for_sort, rating)
                    recommended_items = np.append(recommended_items, i)
        tmp = np.argsort(rating_for_sort)[-limit:]
        ans = recommended_items[tmp]
        ans = np.array(ans, dtype=np.int32)
        return ans

    def recommend_top(self, u, top_x):
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()
        item = {'id': None, 'similar': None}
        list_items = []

        def take_similar(elem):
            return elem['similar']

        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                item['id'] = i
                item['similar'] = rating
                list_items.append(item.copy())

        sorted_items = sorted(list_items, key=take_similar, reverse=True)
        sorted_items.pop(top_x)
        return sorted_items

    def print_recommendation(self):
        """
        print all items which should be recommended for each user
        """
        print('Recommendation: ')
        for u in range(self.n_users):
            recommended_items = self.recommend(u)
            if self.uuCF:
                print(recommended_items, 'for user', u)
            else:
                print('for user(s) : ', recommended_items)
    
data = np.loadtxt("line_2021-07-01_19_09_29_1.dat")

model = CF(data)
model.fit()
print("recomend:")
list_recomend = []
n_users = int(np.max(data[:, 0])) + 1
model.print_list_item()
for i in range(1, 10):
    print("recomend: ", model.recommend(i))

# for i in range(1, n_users):
#     tmp = model.recommend(i)
#     list_recomend.append(tmp)
#     print(f'user {i} : {tmp}')
# list_recomend = np.array(list_recomend)
# print("list_recomend")
# print(list_recomend)
# print(len(list_recomend))
# print("Done collab")

