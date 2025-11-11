import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import pickle
from algorithm.llm_gnn.utils.metrics import Metric
from algorithm.llm_gnn.models.lightgcn import LightGCN
from algorithm.llm_gnn.utils.data_handler import DataHandler, DataHandlerCache

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)


class TrainGCN:
    def __init__(self, investor_features=None, product_features=None):
        self.full_df = None
        default_investor_features = [
            'gender', 'age', 'edu_level', 'University', 'CURRENTIDENTITY',
            'credit', 'weightedbidrate_percent',
            'baddebts_percent', 'user_invest_count'
        ]
        # 默认产品特征
        default_product_features = [
            'total', 'apr_percent', 'term', 'REPAYMENT', 'level',
            'project_invest_count'
        ]
        self.investor_features = investor_features if investor_features is not None else default_investor_features
        self.product_features = product_features if product_features is not None else default_product_features
        print(f"GNN model: LightGCN")
        print(f"Dataset: 拍拍贷")
        self.trn_loader, self.val_loader, self.tst_loader = DataHandler().load_data()  # 获得训练集、验证集、测试集批次
        self.trn_mat, self.val_mat, self.tst_mat = DataHandler().load_mat()  # 获得稀疏张量
        self.trn_adj = DataHandler().create_adjacency_matrix(  # 训练集的邻接矩阵
            f"./data/ppdxy/total_trn.csv"
        )
        with open(f"./data/ppdxy/para_dict.pickle", "rb") as file:
            self.para_dict = pickle.load(file)
        self.user_num = self.para_dict["user_num"]  # 89
        self.item_num = self.para_dict["item_num"]  # 439
        self.metric = Metric()
        self.user_embeds_path = f"./data/ppdxy/emb/user_emb.pkl"
        self.item_embeds_path = f"./data/ppdxy/emb/item_emb.pkl"

    def train(self):
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)

        # Initialize model
        model = LightGCN(
            self.user_num,
            self.item_num,
            self.trn_mat,
            filter_conditions=filter_conditions,
            custom_investor_features=self.investor_features,  # 确保传递了investor_features
            custom_product_features=self.product_features  # 确保传递了product_features
        )
        model = model.to(device)

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)  # Adam优化器适合处理稀疏梯度，用于更新模型参数，学习率设置为0.05
        best_val_value = 0.0  # 在训练过程中跟踪验证集上的最佳表现
        # Train model
        for epoch in range(300):
            total_loss = 0
            model.train()
            for batch in self.trn_loader:
                for i in batch:
                    i = i.to(device)
                optimizer.zero_grad()
                loss = model.cal_loss(batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Evaluation, apply early stop
            model.eval()
            result = self.metric.eval(
                model, self.val_loader, self.para_dict["val_user_nb"]
            )

            val_value = result["recall"].item()
            if val_value > best_val_value:
                patience = 0
                best_val_value = val_value
                recall = result["recall"].item()
                ndcg = result["ndcg"].item()
                precision = result["precision"].item()
                mrr = result["mrr"].item()
                f1 = result["f1"].item()  # Extract F1-score
                # save the user and item embeddings
                user_embeds, item_embeds = model.forward(self.trn_mat)
                with open(self.user_embeds_path, "wb") as file:
                    pickle.dump(user_embeds, file)
                with open(self.item_embeds_path, "wb") as file:
                    pickle.dump(item_embeds, file)

            print(
                f"Epoch {epoch}, Loss: {total_loss:.4f}, Patience: {patience}, Recall: {val_value:.4f}, F1-score: {f1:.4f}"
            )
            if patience >= 40:
                break
            patience += 1
        print("Training finished")
        print(
            f"Best Recall: {recall:.4f}, NDCG: {ndcg:.4f}, Precision: {precision:.4f}, MRR: {mrr:.4f}, F1-score: {f1:.4f}"
        )
        # by jiang 在训练完成后添加保存代码
        # with open('./data/ppd/user_emb.pkl', 'wb') as f:
        #     pickle.dump(user_emb, f)

        # with open('./data/ppd/item_emb.pkl', 'wb') as f:
        #     pickle.dump(item_emb, f)
        torch.save(model.user_emb, 'data/ppdxy/emb/user_emb.pkl')
        torch.save(model.item_emb, 'data/ppdxy/emb/item_emb.pkl')
        print('嵌入文件已更新')  # 添加日志提示


class TrainGNNCache:
    def __init__(self, output_path, data_handler_cache: DataHandlerCache, investor_features=None,
                 product_features=None):
        self.full_df = None
        default_investor_features = [
            'gender', 'age', 'edu_level', 'University', 'CURRENTIDENTITY',
            'credit', 'weightedbidrate_percent',
            'baddebts_percent', 'user_invest_count'
        ]
        # 默认产品特征
        default_product_features = [
            'total', 'apr_percent', 'term', 'REPAYMENT', 'level',
            'project_invest_count'
        ]
        self.investor_features = investor_features if investor_features is not None else default_investor_features
        self.product_features = product_features if product_features is not None else default_product_features
        print(f"GNN model: LightGCN")
        print(f"Dataset: 拍拍贷")
        self.trn_loader, self.val_loader, self.tst_loader = data_handler_cache.load_data()  # 获得训练集、验证集、测试集批次
        self.trn_mat, self.val_mat, self.tst_mat = data_handler_cache.load_mat()  # 获得稀疏张量
        # 训练集的邻接矩阵
        self.trn_adj = DataHandlerCache.create_adjacency_matrix(df=data_handler_cache.train_data,
                                                                user_num=data_handler_cache.user_num,
                                                                item_num=data_handler_cache.item_num)
        self.output_path = output_path
        self.para_dict = data_handler_cache.para_dict
        self.user_num = data_handler_cache.user_num
        self.item_num = data_handler_cache.item_num
        self.metric = Metric(output_path=self.output_path)
        self.user_embeds_path = self.output_path + "/user_emb.pkl"
        self.item_embeds_path = self.output_path + "/item_emb.pkl"
        self.base_data = data_handler_cache.base_data
        self.base_data_yuan = data_handler_cache.base_data_yuan

    def train(self, filter_conditions,data_path):
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)

        # Initialize model
        model = LightGCN(
            num_user=self.user_num,
            num_item=self.item_num,
            adj=self.trn_mat,
            data_path=data_path,
            output_path=self.output_path,
            filter_conditions=filter_conditions,
            custom_investor_features=self.investor_features,  # 确保传递了investor_features
            custom_product_features=self.product_features  # 确保传递了product_features
        )
        model = model.to(device)

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)  # Adam优化器适合处理稀疏梯度，用于更新模型参数，学习率设置为0.05
        best_val_value = 0.0  # 在训练过程中跟踪验证集上的最佳表现
        # Train model
        for epoch in range(300):
            total_loss = 0
            model.train()
            for batch in self.trn_loader:
                for i in batch:
                    i = i.to(device)
                optimizer.zero_grad()
                loss = model.cal_loss(batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Evaluation, apply early stop
            model.eval()
            result = self.metric.eval(
                model=model,
                dataloader=self.val_loader,
                ground_true=self.para_dict["val_user_nb"],
                base_data=self.base_data,
                base_datayuan=self.base_data_yuan
            )

            val_value = result["recall"].item()
            if val_value > best_val_value:
                patience = 0
                best_val_value = val_value
                recall = result["recall"].item()
                ndcg = result["ndcg"].item()
                precision = result["precision"].item()
                mrr = result["mrr"].item()
                f1 = result["f1"].item()  # Extract F1-score
                # save the user and item embeddings
                user_embeds, item_embeds = model.forward(self.trn_mat)
                with open(self.user_embeds_path, "wb") as file:
                    pickle.dump(user_embeds, file)
                with open(self.item_embeds_path, "wb") as file:
                    pickle.dump(item_embeds, file)

            print(
                f"Epoch {epoch}, Loss: {total_loss:.4f}, Patience: {patience}, Recall: {val_value:.4f}, F1-score: {f1:.4f}"
            )
            if patience >= 40:
                break
            patience += 1
        print("Training finished")
        print(
            f"Best Recall: {recall:.4f}, NDCG: {ndcg:.4f}, Precision: {precision:.4f}, MRR: {mrr:.4f}, F1-score: {f1:.4f}"
        )
        # by jiang 在训练完成后添加保存代码
        # with open('./data/ppd/user_emb.pkl', 'wb') as f:
        #     pickle.dump(user_emb, f)

        # with open('./data/ppd/item_emb.pkl', 'wb') as f:
        #     pickle.dump(item_emb, f)
        torch.save(model.user_emb, self.user_embeds_path)
        torch.save(model.item_emb, self.item_embeds_path)
        print('嵌入文件已更新')  # 添加日志提示


if __name__ == "__main__":
    # 定义样本筛选维度
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
    # 可选投资者特征
    custom_investor_features = [
        'gender', 'age', 'edu_level', 'University', 'CURRENTIDENTITY',
        'credit', 'weightedbidrate_percent',
        'baddebts_percent', 'user_invest_count'
    ]
    # 可选产品特征
    custom_product_features = [
        'total', 'apr_percent', 'term', 'REPAYMENT', 'level',
        'project_invest_count'
    ]
    trainer = TrainGCN(investor_features=custom_investor_features,
                       product_features=custom_product_features, )
    trainer.train()
