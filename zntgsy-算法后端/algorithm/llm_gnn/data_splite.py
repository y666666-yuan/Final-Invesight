import pickle

import pandas as pd


class DataSplitter:
    def __init__(self, total_file, filter_conditions=None):
        self.filter_conditions = filter_conditions
        # 读取基础数据 base_data.csv 文件
        self.df_base = pd.read_csv(total_file)
        if self.filter_conditions is not None:
            for feature, value in self.filter_conditions.items():
                if value is not None:
                    self.df_base = self.filter_dataframe(self.df_base, feature, value)
        self.df_base['user'] = self.df_base['userno'].copy()
        self.df_base['item'] = self.df_base['PROJECTNO'].copy()
        new_datayuan = self.df_base[['user', 'item', 'reward']]  # 保留 reward 列以便筛选
        filtered_datayuan = new_datayuan.query('reward == 1')
        print(filtered_datayuan)
        self.df_baseyuan = filtered_datayuan.sort_values(by=['user', 'item'])
        output_file_path = f"./data/ppdxy/data_baseyuan.csv"
        self.df_baseyuan[['user', 'item']].to_csv(output_file_path, index=False)

        self.df_base['user'] = self.df_base['userno'].astype('category').cat.codes
        self.df_base['item'] = self.df_base['PROJECTNO'].astype('category').cat.codes
        # 选择只保留新生成的列
        new_data = self.df_base[['user', 'item', 'reward']]  # 保留 reward 列以便筛选

        # 筛选 reward=1 的行
        filtered_data = new_data.query('reward == 1')

        # 按照 user 列和 item 列进行排序
        self.df_base = filtered_data.sort_values(by=['user', 'item'])

        # 指定要保存的新的 CSV 文件路径
        output_file_path = f"./data/ppdxy/data_base.csv"  # 请将其替换为您想保存的 CSV 文件路径

        # 保存结果到新的 CSV 文件
        self.df_base[['user', 'item']].to_csv(output_file_path, index=False)  # 只保存 user 和 item 列
        self.df = self.df_base

    def filter_dataframe(self, df, feature, value):
        if isinstance(value, (int, float)):
            return df[df[feature] == value]
        elif isinstance(value, dict):
            try:
                max_value = int(value['max'])
                df = df[df[feature] < max_value]
            except Exception as e:
                print(e)
                print(f"\n过滤值错误! {feature}:{value} ")
            try:
                min_value = int(value['min'])
                df = df[df[feature] >= min_value]
            except Exception as e:
                print(e)
                print(f"\n过滤值错误! {feature}:{value} ")
            return df
        elif isinstance(value, list):
            try:
                df = df[df[feature].isin(values=value)]
            except Exception as e:
                print(e)
                print(f"\n过滤值错误! {feature}:{value} ")
            return df
        else:
            return df[df[feature].astype(str) == str(value)]

    def split_data(self):
        # 按用户编号分组
        user_groups = self.df.groupby('user')

        # 创建空的DataFrame存储训练集、验证集、测试集数据
        train_data = pd.DataFrame(columns=self.df.columns)
        val_data = pd.DataFrame(columns=self.df.columns)
        test_data = pd.DataFrame(columns=self.df.columns)

        # 对每个用户的数据进行处理
        for _, group in user_groups:
            # 将该用户的所有样本加入训练集
            train_data = pd.concat([train_data, group])
            # 可以选择将同样的样本也加入验证集和测试集
            val_data = pd.concat([val_data, group])
            test_data = pd.concat([test_data, group])

        # 保存为CSV文件
        train_data.to_csv("./data/ppdxy/total_trn.csv", index=False)
        val_data.to_csv("./data/ppdxy/total_val.csv", index=False)
        test_data.to_csv("./data/ppdxy/total_tst.csv", index=False)

        # 如果有需要生成 para_dict.pickle 的话，可以在这里调用
        self.generate_para_dict(train_data, val_data, test_data)

        return "数据切分完成，已保存为 total_trn.csv, total_val.csv, total_tst.csv"

    def generate_para_dict(self, train_data, val_data, test_data):

        user_num = max(self.df['user']) + 1
        item_num = max(self.df['item']) + 1

        # 创建用户和物品的邻接表
        trn_user_nb = [[] for _ in range(user_num)]
        trn_item_nb = [[] for _ in range(item_num)]
        for _, row in train_data.iterrows():
            trn_user_nb[row["user"]].append(row["item"])
            trn_item_nb[row["item"]].append(row["user"])

        val_user_nb = [[] for _ in range(user_num)]
        val_item_nb = [[] for _ in range(item_num)]
        for _, row in val_data.iterrows():
            val_user_nb[row["user"]].append(row["item"])
            val_item_nb[row["item"]].append(row["user"])

        tst_user_nb = [[] for _ in range(user_num)]
        tst_item_nb = [[] for _ in range(item_num)]
        for _, row in test_data.iterrows():
            tst_user_nb[row["user"]].append(row["item"])
            tst_item_nb[row["item"]].append(row["user"])

        # 构建para_dict字典
        para_dict = {
            "trn_user_nb": trn_user_nb,
            "trn_item_nb": trn_item_nb,
            "val_user_nb": val_user_nb,
            "val_item_nb": val_item_nb,
            "tst_user_nb": tst_user_nb,
            "tst_item_nb": tst_item_nb,
            "user_num": user_num,
            "item_num": item_num,
        }

        # 保存为pickle文件
        with open("./data/ppdxy/para_dict.pickle", "wb") as handle:  # 生成para_dict.pickle文件
            pickle.dump(para_dict, handle)
        print("para_dict saved")


class DataSplitterCache:
    def __init__(self, total_file, filter_conditions=None):
        self.filter_conditions = filter_conditions
        # 读取基础数据 base_data.csv 文件
        self.df_base = pd.read_csv(total_file)
        if self.filter_conditions is not None:
            for feature, value in self.filter_conditions.items():
                if value is not None:
                    self.df_base = self.filter_dataframe(self.df_base, feature, value)
        self.df_base['user'] = self.df_base['userno'].copy()
        self.df_base['item'] = self.df_base['PROJECTNO'].copy()
        new_datayuan = self.df_base[['user', 'item', 'reward']]  # 保留 reward 列以便筛选
        filtered_datayuan = new_datayuan.query('reward == 1')
        print(filtered_datayuan)
        self.df_baseyuan = filtered_datayuan.sort_values(by=['user', 'item'])
        # 取消保存到文件，直接在内存中保存
        self.df_baseyuan = self.df_baseyuan[['user', 'item']]

        self.df_base['user'] = self.df_base['userno'].astype('category').cat.codes
        self.df_base['item'] = self.df_base['PROJECTNO'].astype('category').cat.codes
        # 选择只保留新生成的列
        new_data = self.df_base[['user', 'item', 'reward']]  # 保留 reward 列以便筛选

        # 筛选 reward=1 的行
        filtered_data = new_data.query('reward == 1')

        # 按照 user 列和 item 列进行排序
        self.df_base = filtered_data.sort_values(by=['user', 'item'])

        self.df = self.df_base
        # 保存内存
        self.df_base = self.df_base[['user', 'item']]
        self.cache_data = {}

    def filter_dataframe(self, df, feature, value):
        if isinstance(value, (int, float)):
            return df[df[feature] == value]
        elif isinstance(value, dict):
            try:
                max_value = int(value['max'])
                df = df[df[feature] < max_value]
            except Exception as e:
                print(e)
                print(f"\n过滤值错误! {feature}:{value} ")
            try:
                min_value = int(value['min'])
                df = df[df[feature] >= min_value]
            except Exception as e:
                print(e)
                print(f"\n过滤值错误! {feature}:{value} ")
            return df
        elif isinstance(value, list):
            try:
                df = df[df[feature].isin(values=value)]
            except Exception as e:
                print(e)
                print(f"\n过滤值错误! {feature}:{value} ")
            return df
        else:
            return df[df[feature].astype(str) == str(value)]

    def split_data(self):
        # 按用户编号分组
        user_groups = self.df.groupby('user')

        # 创建空的DataFrame存储训练集、验证集、测试集数据
        train_data = pd.DataFrame(columns=self.df.columns)
        val_data = pd.DataFrame(columns=self.df.columns)
        test_data = pd.DataFrame(columns=self.df.columns)

        # 对每个用户的数据进行处理
        for _, group in user_groups:
            # 将该用户的所有样本加入训练集
            train_data = pd.concat([train_data, group])
            # 可以选择将同样的样本也加入验证集和测试集
            val_data = pd.concat([val_data, group])
            test_data = pd.concat([test_data, group])

        # 保存为CSV文件
        # train_data.to_csv("./data/ppdxy/total_trn.csv", index=False)
        # val_data.to_csv("./data/ppdxy/total_val.csv", index=False)
        # test_data.to_csv("./data/ppdxy/total_tst.csv", index=False)
        self.cache_data['train_data'] = train_data
        self.cache_data['val_data'] = val_data
        self.cache_data['test_data'] = test_data

        # 如果有需要生成 para_dict.pickle 的话，可以在这里调用
        self.generate_para_dict(train_data, val_data, test_data)

        return "数据切分完成，已保存为 total_trn.csv, total_val.csv, total_tst.csv"

    def generate_para_dict(self, train_data, val_data, test_data):

        user_num = max(self.df['user']) + 1
        item_num = max(self.df['item']) + 1

        # 创建用户和物品的邻接表
        trn_user_nb = [[] for _ in range(user_num)]
        trn_item_nb = [[] for _ in range(item_num)]
        for _, row in train_data.iterrows():
            trn_user_nb[row["user"]].append(row["item"])
            trn_item_nb[row["item"]].append(row["user"])

        val_user_nb = [[] for _ in range(user_num)]
        val_item_nb = [[] for _ in range(item_num)]
        for _, row in val_data.iterrows():
            val_user_nb[row["user"]].append(row["item"])
            val_item_nb[row["item"]].append(row["user"])

        tst_user_nb = [[] for _ in range(user_num)]
        tst_item_nb = [[] for _ in range(item_num)]
        for _, row in test_data.iterrows():
            tst_user_nb[row["user"]].append(row["item"])
            tst_item_nb[row["item"]].append(row["user"])

        # 构建para_dict字典
        para_dict = {
            "trn_user_nb": trn_user_nb,
            "trn_item_nb": trn_item_nb,
            "val_user_nb": val_user_nb,
            "val_item_nb": val_item_nb,
            "tst_user_nb": tst_user_nb,
            "tst_item_nb": tst_item_nb,
            "user_num": user_num,
            "item_num": item_num,
        }

        # 保存为pickle文件
        # with open("./data/ppdxy/para_dict.pickle", "wb") as handle:  # 生成para_dict.pickle文件
        #     pickle.dump(para_dict, handle)
        # print("para_dict saved")
        self.cache_data['para_dict'] = para_dict


if __name__ == "__main__":
    filter_conditions = {
        'gender': None,
        'age': None,
        'edu_level': None,
        'CURRENTIDENTITY': None,
        'user_invest_count': None,
        'total': None,
        'apr_percent': None,
        'term': None,
        'REPAYMENT': None,
        'level': None,
        'project_invest_count': None
    }
    # 这是作为独立脚本运行时的入口点
    # 设置基础数据文件的默认路径
    total_file = f"./data/Data/面板数据.csv"
    splitter = DataSplitter(total_file, filter_conditions=filter_conditions)
    result = splitter.split_data()
    print(result)
