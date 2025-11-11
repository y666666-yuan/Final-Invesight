import pickle
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import coo_matrix, csr_matrix
from torch.utils.data import Dataset, DataLoader


class Dataset(Dataset):
    def __init__(self, user_list, item_list):
        self.user_list = user_list
        self.item_list = item_list

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, index):
        return self.user_list[index], self.item_list[index]


class TripleData(Dataset):
    def __init__(self, user_list, pos_item_list, neg_item_list):
        self.user_list = user_list
        self.pos_item_list = pos_item_list
        self.neg_item_list = neg_item_list

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, index):
        return (
            self.user_list[index],
            self.pos_item_list[index],
            self.neg_item_list[index],
        )


class DataHandler:
    def __init__(self):
        predir = f"./data/ppdxy/"
        with open(f"./data/ppdxy/para_dict.pickle", "rb") as file:
            self.para_dict = pickle.load(file)
        self.user_num = self.para_dict["user_num"]  # 89
        self.item_num = self.para_dict["item_num"]  # 439
        self.trn_path = predir + "total_trn.csv"
        self.val_path = predir + "total_val.csv"
        self.tst_path = predir + "total_tst.csv"

    def load_csv(self, file_path):
        """return user, item as list"""
        df = pd.read_csv(file_path)
        user_list = df["user"].tolist()
        item_list = df["item"].tolist()
        return user_list, item_list

    def load_csv_with_negative_sampling(self, file_path):
        """return user, item as list"""
        df = pd.read_csv(file_path)
        user_list = df["user"].tolist()
        pos_item_list = df["item"].tolist()

        all_items = df["item"].unique()
        user_interacted_items = df.groupby("user")["item"].apply(set).to_dict()

        neg_item_list = []
        for index, row in df.iterrows():
            user = row["user"]
            user_items = user_interacted_items[user]
            negative_item = random.choice(all_items)
            while negative_item in user_items:
                negative_item = random.choice(all_items)
            neg_item_list.append(negative_item)
        return user_list, pos_item_list, neg_item_list

    def create_adjacency_matrix(self, file):
        user_list, item_list = self.load_csv(file)
        # Create coo matrix
        adj_matrix = coo_matrix(
            (np.ones(len(user_list)), (user_list, item_list)),
            shape=(self.user_num, self.item_num),
        )
        return adj_matrix

    def make_torch_adj(self, adj_matrix):
        a = csr_matrix((self.user_num, self.user_num))
        b = csr_matrix((self.item_num, self.item_num))

        mat = sp.vstack(
            [sp.hstack([a, adj_matrix]), sp.hstack([adj_matrix.transpose(), b])]
        )
        mat = (mat != 0) * 1.0

        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        mat = mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

        # make torch tensor
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape)

    def load_data(self):
        # load training triple batch 训练三元组批次
        user_list, pos_item_list, neg_item_list = self.load_csv_with_negative_sampling(self.trn_path)
        trn_dataset = TripleData(user_list, pos_item_list, neg_item_list)
        trn_loader = DataLoader(trn_dataset, batch_size=1024, shuffle=True)

        # load validation batch 验证集批次
        user_list, item_list = self.load_csv(self.val_path)
        val_dataset = Dataset(user_list, item_list)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=True)

        # load testing batch 测试集批次
        user_list, item_list = self.load_csv(self.tst_path)
        tst_dataset = Dataset(user_list, item_list)
        tst_loader = DataLoader(tst_dataset, batch_size=1024, shuffle=True)

        return trn_loader, val_loader, tst_loader

    def load_mat(self):
        trn_mat = self.create_adjacency_matrix(self.trn_path)
        val_mat = self.create_adjacency_matrix(self.val_path)
        tst_mat = self.create_adjacency_matrix(self.tst_path)

        trn_mat = self.make_torch_adj(trn_mat)
        val_mat = self.make_torch_adj(val_mat)
        tst_mat = self.make_torch_adj(tst_mat)
        return trn_mat, val_mat, tst_mat


class DataHandlerCache:
    def __init__(self, para_dict, train_data, val_data, test_data, base_data, base_data_yuan):
        self.para_dict = para_dict
        self.user_num = self.para_dict["user_num"]
        self.item_num = self.para_dict["item_num"]
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.base_data = base_data
        self.base_data_yuan = base_data_yuan

    def load_csv(self, file_path):
        """return user, item as list"""
        df = pd.read_csv(file_path)
        user_list = df["user"].tolist()
        item_list = df["item"].tolist()
        return user_list, item_list

    def load_csv_with_negative_sampling(self):
        """
        load training triple batch 训练三元组批次
        return user, item as list
        """

        user_list = self.train_data["user"].tolist()
        pos_item_list = self.train_data["item"].tolist()

        all_items = self.train_data["item"].unique()
        user_interacted_items = self.train_data.groupby("user")["item"].apply(set).to_dict()

        neg_item_list = []
        for index, row in self.train_data.iterrows():
            user = row["user"]
            user_items = user_interacted_items[user]
            negative_item = random.choice(all_items.tolist())
            while negative_item in user_items:
                negative_item = random.choice(all_items.tolist())
            neg_item_list.append(negative_item)
        return user_list, pos_item_list, neg_item_list

    @staticmethod
    def create_adjacency_matrix(df, user_num, item_num):
        user_list = df["user"].tolist()
        item_list = df["item"].tolist()
        # Create coo matrix
        adj_matrix = coo_matrix(
            (np.ones(len(user_list)), (user_list, item_list)),
            shape=(user_num, item_num),
        )
        return adj_matrix

    @staticmethod
    def make_torch_adj(user_num, item_num, adj_matrix):
        a = csr_matrix((user_num, user_num))
        b = csr_matrix((item_num, item_num))

        mat = sp.vstack(
            [sp.hstack([a, adj_matrix]), sp.hstack([adj_matrix.transpose(), b])]
        )
        mat = (mat != 0) * 1.0

        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        mat = mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

        # make torch tensor
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape)

    def load_data(self):
        # load training triple batch 训练三元组批次
        user_list, pos_item_list, neg_item_list = self.load_csv_with_negative_sampling()
        trn_dataset = TripleData(user_list, pos_item_list, neg_item_list)
        trn_loader = DataLoader(trn_dataset, batch_size=1024, shuffle=True)

        # load validation batch 验证集批次
        user_list = self.val_data["user"].tolist()
        item_list = self.val_data["item"].tolist()

        val_dataset = Dataset(user_list, item_list)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=True)

        # load testing batch 测试集批次
        user_list = self.test_data["user"].tolist()
        item_list = self.test_data["item"].tolist()
        tst_dataset = Dataset(user_list, item_list)
        tst_loader = DataLoader(tst_dataset, batch_size=1024, shuffle=True)

        return trn_loader, val_loader, tst_loader

    def load_mat(self):
        trn_mat = self.create_adjacency_matrix(self.train_data, self.user_num, self.item_num)
        val_mat = self.create_adjacency_matrix(self.val_data, self.user_num, self.item_num)
        tst_mat = self.create_adjacency_matrix(self.test_data, self.user_num, self.item_num)

        trn_mat = self.make_torch_adj(self.user_num, self.item_num, trn_mat)
        val_mat = self.make_torch_adj(self.user_num, self.item_num, val_mat)
        tst_mat = self.make_torch_adj(self.user_num, self.item_num, tst_mat)
        return trn_mat, val_mat, tst_mat
