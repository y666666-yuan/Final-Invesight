import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
from sklearn.preprocessing import StandardScaler


class LightGCN(nn.Module):
    def __init__(self, num_user, num_item, adj, data_path, output_path, filter_conditions=None,
                 custom_investor_features=None,
                 custom_product_features=None):
        super(LightGCN, self).__init__()
        # 明确初始化output_path属性
        self.output_path = output_path

        # 确保目录存在
        os.makedirs(self.output_path, exist_ok=True)
        # 保存特征列表到类属性
        self.filter_conditions = filter_conditions
        self.custom_investor_features = custom_investor_features
        self.custom_product_features = custom_product_features
        self.num_user = num_user
        self.num_item = num_item
        self.embedding_dim = 64
        self.n_layers = 4
        self.adj = adj

        # 加载并预处理数据
        self.full_df = self.load_and_preprocess_data(
            data_path=data_path,
            filter_conditions=None
        )
        # 使用自定义特征或默认特征
        if custom_investor_features:
            user_features = self.full_df[custom_investor_features][:self.num_user].values
        else:
            user_features = self.full_df[['age_weighted', 'totalcreditscore_weighted', 'gender_weighted']][
                            :self.num_user].values

        if custom_product_features:
            item_features = self.full_df[custom_product_features][:self.num_item].values
        else:
            item_features = self.full_df[['total_weighted', 'term1_weighted', 'interate_weighted']][
                            :self.num_item].values

        # 转为Tensor并初始化网络层
        self.user_features = torch.tensor(user_features, dtype=torch.float32)
        self.item_features = torch.tensor(item_features, dtype=torch.float32)

        self.embeds_list = []  # by jiang 显式初始化嵌入列表
        self.user_emb = None  # by jiang添加类属性存储最终嵌入
        self.item_emb = None
        # 初始化嵌入层
        self.user_embedding = nn.Embedding(num_user, self.embedding_dim)
        self.item_embedding = nn.Embedding(num_item, self.embedding_dim)

        # # Initialize user and item embeddings
        # self.user_embeds = nn.Parameter(
        #     nn.init.xavier_uniform_(torch.empty(self.num_user, self.embedding_dim))
        # )

        # self.item_embeds = nn.Parameter(
        #     nn.init.xavier_uniform_(torch.empty(self.num_item, self.embedding_dim))
        # )

        # 转为Tensor并移动到设备上
        self.user_features = torch.tensor(user_features, dtype=torch.float32)
        self.item_features = torch.tensor(item_features, dtype=torch.float32)

        # 全连接层将用户特征和物品特征映射到64维
        self.user_fc = nn.Linear(self.user_features.shape[1], self.embedding_dim)
        self.item_fc = nn.Linear(self.item_features.shape[1], self.embedding_dim)

        a = 0.02

        # 权重初始化，确保输出较小
        init.uniform_(self.user_fc.weight, -a, a)  # 使用较小的初始化范围
        init.uniform_(self.item_fc.weight, -a, a)

        # 使用Tanh激活函数确保输出值小于1
        self.user_embeds = torch.tanh(self.user_fc(self.user_features))  # 输出在[-1, 1]之间
        self.item_embeds = torch.tanh(self.item_fc(self.item_features))  # 输出在[-1, 1]之间

        # 或者使用Sigmoid确保输出值在(0, 1)之间
        # self.user_embeds = torch.sigmoid(self.user_fc(self.user_features))  # 输出在(0, 1)之间
        # self.item_embeds = torch.sigmoid(self.item_fc(self.item_features))  # 输出在(0, 1)之间
        n = 0.00001
        # 如果你想要输出值特别小（例如 < 0.1），可以对嵌入向量再进行缩放
        self.user_embeds = self.user_embeds * n
        self.item_embeds = self.item_embeds * n

        self.user_embeds = nn.Parameter(self.user_embeds)
        self.item_embeds = nn.Parameter(self.item_embeds)

        print(f"user_embeds: {self.user_embeds}")
        print(f"item_embeds: {self.user_embeds}")

        # Check if user_embeds and item_embeds are tensors
        print(f"user_embeds is tensor: {isinstance(self.user_embeds, torch.Tensor)}")
        print(f"item_embeds is tensor: {isinstance(self.item_embeds, torch.Tensor)}")

        # Print the attributes of user_embeds and item_embeds
        print(f"user_embeds shape: {self.user_embeds.shape}, requires_grad: {self.user_embeds.requires_grad}")
        print(f"item_embeds shape: {self.item_embeds.shape}, requires_grad: {self.item_embeds.requires_grad}")
        self.user_embeds1 = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.num_user, self.embedding_dim))
        )
        self.item_embeds1 = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.num_item, self.embedding_dim))
        )  # Xavier初始化有助于防止梯度消失/爆炸问题
        self.user_embeds = nn.Parameter(self.user_embeds + self.user_embeds1)  # 参数相加操作可以融合不同来源的特征信息
        self.item_embeds = nn.Parameter(self.item_embeds + self.item_embeds1)

    def forward(self, adj):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("using device:", device)
        user_emb = self.user_embedding.weight.to(device)
        item_emb = self.item_embedding.weight.to(device)

        embeds_list = [torch.cat([user_emb, item_emb], dim=0)]
        # 替换原embedding_dict的用法
        adj = adj.to(device)

        for _ in range(self.n_layers):
            new_embeds = torch.spmm(adj, embeds_list[-1])
            embeds_list.append(new_embeds)

        return embeds_list[-1][:self.num_user], embeds_list[-1][self.num_user:]

    def cal_loss(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]

        # Calculate the difference between the positive and negative item predictions
        pos_scores = torch.sum(anc_embeds * pos_embeds, dim=1)
        neg_scores = torch.sum(anc_embeds * neg_embeds, dim=1)
        diff_scores = pos_scores - neg_scores

        # Compute the BPR loss
        # print(f"diff_scores range: {diff_scores.min().item()} to {diff_scores.max().item()}")
        loss = -torch.mean(torch.log(torch.sigmoid(diff_scores)))

        # Regularization term (optional)
        reg_loss = 1e-6 * (
                anc_embeds.norm(2).pow(2)
                + pos_embeds.norm(2).pow(2)
                + neg_embeds.norm(2).pow(2)
        )

        return loss + reg_loss

    def full_predict(self, batch_data):
        # 调用forward获取最新嵌入
        user_embeds, item_embeds = self.forward(self.adj)

        # 保存嵌入到类属性（可选）
        self.user_emb, self.item_emb = user_embeds, item_embeds

        # 计算预测结果
        users, _ = batch_data
        pck_user_embeds = user_embeds[users]
        full_preds = pck_user_embeds @ item_embeds.T  # 使用正确的矩阵乘法运算符

        return full_preds  # 保持单一返回类型

    def load_and_preprocess_data(self, data_path, filter_conditions=None):
        """加载并预处理数据，返回合并后的DataFrame和映射字典"""
        if not os.path.exists(data_path):
            print(f"错误：指定的数据路径 {data_path} 不存在。")
            return None
        csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
        if len(csv_files) != 1:
            print(f"错误：路径 {data_path} 下应只有一个 .csv 文件，当前找到 {len(csv_files)} 个。")
            return None
        csv_file = csv_files[0]

        try:
            self.full_df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"读取文件 {csv_file} 时出错：{e}")
            return None

        if filter_conditions is None:
            filter_conditions = {}

        def filter_dataframe(df, feature, value):
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

        # 根据用户输入的特征进行数据筛选
        for feature, value in filter_conditions.items():
            if value is None:
                continue
            else:
                self.full_df = filter_dataframe(self.full_df, feature, value)

        # 打印数据信息
        print("筛选出来的标准化前的数据集信息：")
        self.full_df.info()
        print("标准化前的数据集前几行：")
        print(self.full_df.head().to_csv(sep='\t', na_rep='nan'))

        # 输出处理后的 DataFrame 为 CSV 文件到 output 文件夹
        output_csv_path = os.path.join(self.output_path, "processed_data.csv")
        self.full_df.to_csv(output_csv_path, index=False)
        print(f"处理后的数据集已保存到 {output_csv_path}")

        # 修改需要标准化的连续特征
        cont_features = [
            'age', 'borrowingcredit', 'loancredit', 'weightedbidrate_percent',
            'baddebts_percent', 'user_invest_count', 'total', 'apr_percent',
            'term', 'level', 'project_invest_count'
        ]
        scaler = StandardScaler()
        self.full_df[cont_features] = scaler.fit_transform(self.full_df[cont_features])
        # 计算 credit 特征
        self.full_df['credit'] = (self.full_df['loancredit'] + self.full_df['borrowingcredit']) / 2

        # 生成映射关系
        self.full_df['investor_id'] = self.full_df['userno'].astype('category').cat.codes
        self.full_df['product_id'] = self.full_df['PROJECTNO'].astype('category').cat.codes

        # 创建双向映射字典
        self.id_to_userno = self.full_df[['investor_id', 'userno']].drop_duplicates().set_index('investor_id')[
            'userno'].to_dict()
        self.userno_to_id = {v: k for k, v in self.id_to_userno.items()}

        return self.full_df
