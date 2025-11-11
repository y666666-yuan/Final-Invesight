import math
import os
import time
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import cholesky, solve_triangular
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from core import sys
import json
import torch
# 在run_cmab方法中使用多线程处理用户
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

ALGORITHM_TYPE = 'cmab'


class CMABRunner:
    def __init__(self, investor_features=None, product_features=None, user_id=None,
                 alpha=0.7, gamma=0.0009, lambd=0.0009, delta=0.0009,
                 m=5, T=1000, init_theta=20.0, output_suffix=""):
        self.seed = 42
        np.random.seed(self.seed)
        self.full_df = None
        default_investor_features = [
            'gender', 'age', 'edu_level', 'University', 'CURRENTIDENTITY',
            'credit', 'weightedbidrate_percent', 'baddebts_percent', 'user_invest_count'
        ]
        default_product_features = [
            'total', 'apr_percent', 'term', 'REPAYMENT', 'level', 'project_invest_count'
        ]
        self.investor_features = investor_features or default_investor_features
        self.product_features = product_features or default_product_features

        # 创建带时间戳的结果路径
        # self.output_path = sys.get_file_path("data/task_result", str(user_id) + '_' + ALGORITHM_TYPE,
        #                                      str(int(time.time())))
        #
        # if not os.path.exists(self.output_path):
        #     os.makedirs(self.output_path)
        timestamp = str(int(time.time()))
        self.output_path = sys.get_file_path(sys.BASE_TASK_PATH,
                                             str(sys.ALGORITHM_TYPE_CONSTRAINT),
                                             f"{user_id}_{sys.ALGORITHM_CONSTRAINT_NAME_CMAB}",
                                             timestamp)
        os.makedirs(self.output_path, exist_ok=True)
        # 算法超参数配置
        self.alpha = alpha  # 探索系数
        self.gamma = gamma  # q的步长
        self.lambd = lambd  # Q的步长
        self.delta = delta  # theta的步长
        self.m = m  # 每轮推荐数
        self.T = T  # 总训练轮数
        self.init_theta = init_theta  # theta初始值
        # 设备选择：CUDA 优先，回退 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"[CMAB] 使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[CMAB] 使用 CPU")

    def manual_equal_size_binning(self, data_series, q=4):
        """等额分箱函数"""
        sorted_indices = np.argsort(data_series)
        n = len(data_series)
        bins = np.empty(n, dtype=int)
        bin_size = n // q
        remainder = n % q

        start = 0
        for i in range(q):
            end = start + bin_size + (1 if i < remainder else 0)
            bins[sorted_indices[start:end]] = i
            start = end
        return bins

    def load_and_preprocess(self, data_path, filter_conditions=None):
        if not os.path.exists(data_path):
            print(f"错误：指定的数据路径 {data_path} 不存在。")
            return None
        csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
        if len(csv_files) != 1:
            print(f"错误：路径 {data_path} 下应只有一个 .csv 文件，当前找到 {len(csv_files)} 个。")
            return None
        csv_file = csv_files[0]
        self.full_df = pd.read_csv(csv_file)

        if filter_conditions:
            for feature, value in filter_conditions.items():
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    self.full_df = self.full_df[self.full_df[feature] == value]
                elif isinstance(value, dict):
                    if 'max' in value:
                        self.full_df = self.full_df[self.full_df[feature] < int(value['max'])]
                    if 'min' in value:
                        self.full_df = self.full_df[self.full_df[feature] >= int(value['min'])]
                elif isinstance(value, list):
                    self.full_df = self.full_df[self.full_df[feature].isin(value)]
                else:
                    self.full_df = self.full_df[self.full_df[feature].astype(str) == str(value)]

        # 全局分箱（对连续字段按全数据的四分位进行分箱，生成 *_group 列，取值 1..4）
        cont_group_fields = ['apr_percent', 'term', 'total', 'project_invest_count']
        for f in cont_group_fields:
            if f in self.full_df.columns:
                # 使用百分位排名然后按边界分箱，避免 pd.qcut 在重复值很多时报错
                pct = self.full_df[f].rank(method='average', pct=True, na_option='keep')
                self.full_df[f + '_group'] = pd.cut(pct, bins=[0.0, 0.25, 0.5, 0.75, 1.0],
                                                    labels=[1, 2, 3, 4], include_lowest=True).astype('Int64')

        # 定义所有连续特征（包含投资者和产品特征）并标准化数值特征（用于建模）
        cont_features = [
            'age', 'total', 'apr_percent', 'term',
            'project_invest_count', 'user_invest_count',
            'weightedbidrate_percent', 'baddebts_percent'
        ]
        # 仅对存在的列进行标准化，避免 KeyError
        cont_existing = [c for c in cont_features if c in self.full_df.columns]
        if cont_existing:
            scaler = StandardScaler()
            self.full_df[cont_existing] = scaler.fit_transform(self.full_df[cont_existing])

        # 保存处理后的数据到 output_path（包含 *_group 列）
        raw_data_path = os.path.join(self.output_path, "processed_data.csv")
        self.full_df.to_csv(raw_data_path, index=False)
        print(f"处理后的数据集已保存到 {raw_data_path}")

        # 生成映射关系
        self.full_df['investor_id'] = self.full_df['userno'].astype('category').cat.codes
        self.id_to_userno = self.full_df[['investor_id', 'userno']].drop_duplicates() \
            .set_index('investor_id')['userno'].to_dict()

        return self.full_df

    class RoboAdvisor:
        def __init__(self, alpha, d, K, m, T, gamma, lambd, delta, theta, device):
            self.alpha = alpha
            self.d = d
            self.K = K
            self.m = m
            self.T = T
            self.theta = float(theta)
            self.gamma = float(gamma)
            self.lambd = float(lambd)
            self.delta = float(delta)
            self.device = device
            # 参数放到 device（float32）
            self.A = torch.eye(d, device=self.device, dtype=torch.float32)
            self.b = torch.zeros(d, device=self.device, dtype=torch.float32)
            self.beta = torch.zeros(d, device=self.device, dtype=torch.float32)
            self.Q = torch.tensor([0.0], device=self.device, dtype=torch.float32)
            # 初始化 q：K 维，选 m 个置 1
            self.q = torch.zeros(K, device=self.device, dtype=torch.float32)
            idx = torch.randperm(K, device=self.device)[:m]
            self.q[idx] = 1.0
            # 参数校验
            assert 0 < alpha <= 1, "探索系数alpha应在(0,1]范围内"
            assert m > 0, "推荐数量m应大于0"

        def recommend(self, X_features, constraint_feature):
            # 维度校验
            assert X_features.shape[0] == len(constraint_feature), "特征矩阵与约束向量维度不匹配"
            # 转为 torch 张量（K, d）和（K,）
            X = torch.as_tensor(X_features, device=self.device, dtype=torch.float32)
            c = torch.as_tensor(constraint_feature, device=self.device, dtype=torch.float32)
            # UCB 计算：||L^{-1} x||，L 是 A 的 Cholesky 分解
            try:
                L = torch.linalg.cholesky(self.A)                     # (d,d)
                L_inv_X = torch.linalg.solve_triangular(L, X.t(), upper=False)  # (d,K)
                ucb = self.alpha * torch.sqrt(torch.sum(L_inv_X ** 2, dim=0))   # (K,)
                r_hat = X.matmul(self.beta) + ucb                                 # (K,)
            except RuntimeError:
                r_hat = X.matmul(self.beta)
            # 梯度并更新 q（投影到可行域）
            grad_L = self.theta * r_hat - self.Q[0] * c
            q_new = self.q - self.gamma * grad_L
            self.q = self.project_C(q_new)
            # 选 Top-m
            top_idx = torch.topk(self.q, self.m, largest=True).indices
            return top_idx.detach().cpu().numpy(), r_hat.detach().cpu().numpy()

        def project_C(self, q_new: torch.Tensor) -> torch.Tensor:
            q_clipped = torch.clamp(q_new, 0.0, 1.0)
            if torch.sum(q_clipped) <= self.m + 1e-6:
                return q_clipped
            # 置顶 m 个为 1，其余 0（与原逻辑一致）
            idx = torch.topk(q_clipped, self.m, largest=True).indices
            mask = torch.zeros_like(q_clipped)
            mask[idx] = 1.0
            return mask

    class MABTSLRRecommender:
        def __init__(self, output_path: str, iterations=100, opt_method='L-BFGS-B', opt_options=None,
                     investor_features=None, product_features=None):
            self.full_df = None
            self.feature_cols = None
            self.m = None
            self.q = None
            self.iterations = iterations
            self.opt_method = opt_method
            self.opt_options = opt_options if opt_options is not None else {}
            self.id_to_userno = {}
            self.userno_to_id = {}
            self.investor_features = investor_features if investor_features is not None else []
            self.product_features = product_features if product_features is not None else []

            self.processed_data = os.path.join(output_path, "processed_data.csv")
            self.output_path = os.path.join(output_path, "mabtslr")
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)

        def load_and_preprocess(self):
            """从 processed_data.csv 加载并做最小预处理（设置 feature_cols、映射等）"""
            try:
                self.full_df = pd.read_csv(self.processed_data)
            except Exception as e:
                print(f"读取 {self.processed_data} 失败: {e}")
                return None, None

            # 定义特征列（投资者特征 + 产品特征）
            self.feature_cols = list(self.investor_features) + list(self.product_features)

            # 确保特征列存在并为数值类型
            for c in self.feature_cols:
                if c not in self.full_df.columns:
                    self.full_df[c] = 0.0
            self.full_df[self.feature_cols] = self.full_df[self.feature_cols].astype(float)

            # 生成映射关系
            if 'userno' in self.full_df.columns:
                self.full_df['investor_id'] = self.full_df['userno'].astype('category').cat.codes
                self.id_to_userno = self.full_df[['investor_id', 'userno']].drop_duplicates().set_index('investor_id')\
                    ['userno'].to_dict()
                self.userno_to_id = {v: k for k, v in self.id_to_userno.items()}
            else:
                self.full_df['investor_id'] = 0
                self.id_to_userno = {0: 0}
                self.userno_to_id = {0: 0}

            # 初始化参数
            self.m = np.zeros(len(self.feature_cols))
            self.q = np.ones(len(self.feature_cols))

            # 保存预处理数据副本（保持兼容）
            output_csv_path = os.path.join(self.output_path, "processed_data.csv")
            self.full_df.to_csv(output_csv_path, index=False)
            print(f"处理后的数据集已保存到 {output_csv_path}")

            return self.full_df, self.feature_cols

        def update_parameters(self, X, y):
            """更新模型参数（同原逻辑）"""
            # 计算概率向量
            p = 1 / (1 + np.exp(-X.dot(self.m)))
            p = p.reshape(-1, 1)

            # 定义优化目标函数
            def loss(u):
                reg_term = 0.5 * np.sum(self.q * (u - self.m) ** 2)
                log_loss = np.sum(np.log(1 + np.exp(-y * X.dot(u))))
                return reg_term + log_loss

            # 约束优化（m_i >= 0）
            bounds = [(0, None)] * len(self.feature_cols)
            res = minimize(loss, self.m, method=self.opt_method, bounds=bounds, options=self.opt_options)
            self.m = res.x

            # 更新 q（广播正确）
            self.q += np.sum(X ** 2 * p * (1 - p), axis=0)

        def train_model(self, metrics_to_calculate):
            """训练模型（迭代更新 m）"""
            if self.full_df is None or self.feature_cols is None:
                print("请先加载并预处理数据。")
                return None, None, {}

            X = self.full_df[self.feature_cols].values
            y = self.full_df['reward'].values if 'reward' in self.full_df.columns else np.zeros(X.shape[0])

            for _ in range(self.iterations):
                self.update_parameters(X, y)

            # 模型评估
            probas = 1 / (1 + np.exp(-self.full_df[self.feature_cols].values.dot(self.m)))
            preds = (probas > 0.5).astype(int)

            # 计算评估指标
            metrics = {}
            if 'accuracy' in metrics_to_calculate:
                metrics['accuracy'] = float(accuracy_score(y, preds))
            if 'precision' in metrics_to_calculate:
                metrics['precision'] = float(precision_score(y, preds, zero_division=0))
            if 'recall' in metrics_to_calculate:
                metrics['recall'] = float(recall_score(y, preds, zero_division=0))
            if 'f1' in metrics_to_calculate:
                metrics['f1'] = float(f1_score(y, preds, zero_division=0))
            if 'auc' in metrics_to_calculate:
                try:
                    metrics['auc'] = float(roc_auc_score(y, probas))
                except Exception:
                    metrics['auc'] = 0.0

            print("\n模型评估指标:")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

            return self.m, self.q, metrics

        def get_priority_vector(self):
            """获取归一化的优先级向量"""
            priority = self.m.copy()
            priority[priority < 0] = 0
            return priority / (priority.sum() + 1e-8)

        def generate_recommendations(self, target_investor_id):
            """基于目标用户特征生成个性化推荐（与单独的 mabtslr.py 保持一致）"""
            try:
                target_userno = self.id_to_userno[target_investor_id]
            except Exception:
                # 若映射失败，尝试直接使用 id 作为 userno 字符串
                target_userno = target_investor_id

            # 目标用户特征均值
            target_mask = self.full_df['userno'] == target_userno
            target_features = self.full_df.loc[target_mask, self.feature_cols].mean(axis=0).values

            # 预计算所有用户的平均特征向量
            user_priorities = {
                user: self.full_df.loc[self.full_df['userno'] == user, self.feature_cols].mean(axis=0).values
                for user in self.full_df['userno'].unique()
            }

            # 个性化 priority：全局 m 与用户特征结合
            global_pref = np.maximum(0.0, self.m)
            if np.linalg.norm(global_pref) > 1e-12:
                personalized = global_pref * target_features
            else:
                personalized = target_features.copy()
            if np.linalg.norm(personalized) < 1e-12:
                personalized = np.ones_like(personalized)
            target_priority = personalized / (np.linalg.norm(personalized) + 1e-12)

            # 以目标用户特征为基础计算相似度
            similarities = {}
            tf = target_features
            tfnorm = np.linalg.norm(tf) + 1e-12
            for user, features in user_priorities.items():
                if user == target_userno:
                    continue
                fnorm = np.linalg.norm(features) + 1e-12
                similarities[user] = float(np.dot(tf, features) / (tfnorm * fnorm))

            top_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]

            # 计算产品得分
            product_scores = {}
            all_products = self.full_df['PROJECTNO'].unique()

            for product in all_products:
                pf_rows = self.full_df[self.full_df['PROJECTNO'] == product][self.feature_cols]
                if pf_rows.empty:
                    continue
                product_features = pf_rows.iloc[0].values
                base_score = float(np.dot(target_priority, product_features))

                total_weight = sum(max(0.0, sim) for _, sim in top_users)
                if total_weight < 1e-12:
                    score = base_score
                else:
                    score = 0.0
                    for user, sim in top_users:
                        user_purchased = self.full_df[
                            (self.full_df['userno'] == user) &
                            (self.full_df['PROJECTNO'] == product) &
                            (self.full_df.get('reward', 0) == 1)
                        ]
                        if not user_purchased.empty:
                            score += (max(0.0, sim) / total_weight) * base_score
                product_scores[product] = score

            sorted_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)[:5]

            # 表1：Top-5相似投资者
            table1 = pd.DataFrame({
                'Userno': [u[0] for u in top_users],
                'Similarity': [u[1] for u in top_users]
            })

            # 表2：Top-5推荐产品
            table2 = pd.DataFrame({
                'Projectno': [p[0] for p in sorted_products],
                'Score': [p[1] for p in sorted_products]
            })

            user_weights = target_priority * target_features
            user_weights = user_weights / (user_weights.sum() + 1e-8)
            feature_names = self.feature_cols
            table3 = pd.DataFrame({
                'Feature': feature_names,
                'Contribution': user_weights
            }).sort_values('Contribution', ascending=False)

            # added by jiang for 筛选非零特征贡献值的特征
            non_zero_table3 = table3[table3['Contribution'] > 0]
            if non_zero_table3.empty:
                # 若所有特征贡献值都为零，创建一个默认的 DataFrame
                non_zero_table3 = pd.DataFrame({
                    'Feature': ['No significant feature'],
                    'Contribution': [1.0]
                })

            sample_count = len(self.full_df[self.full_df['investor_id'] == target_investor_id]) \
                if 'investor_id' in self.full_df.columns else len(self.full_df[self.full_df['userno'] == target_userno])
            return table1, table2, non_zero_table3, target_userno, sample_count

        # 在MABTSLRRecommender中同样使用多线程处理用户推荐
        def generate_recommendations_parallel(self, target_investor_ids):
            """并行生成推荐结果（提交单参方法以匹配 generate_recommendations 签名）"""
            print("开始并行触发（ThreadPoolExecutor）")
            results = []
            with ThreadPoolExecutor(max_workers=max(1, os.cpu_count() - 1)) as executor:
                futures = {executor.submit(self.generate_recommendations, int(tid)): tid for tid in target_investor_ids}
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        print(f"生成推荐时发生错误: {str(e)}")
            return results

        def run_algorithm(self, metrics_to_calculate=None):
            """入口：加载数据、训练、并行生成推荐并保存结果"""
            if metrics_to_calculate is None:
                metrics_to_calculate = ['accuracy', 'precision', 'recall', 'f1', 'auc']

            # 加载并预处理
            loaded = self.load_and_preprocess()
            if loaded is None:
                return None, "数据加载或预处理失败"
            # 训练
            self.m, self.q, metrics = self.train_model(metrics_to_calculate)
            print("模型训练完成")

            # 生成所有用户推荐
            all_investor_ids = self.full_df['investor_id'].unique()
            results = self.generate_recommendations_parallel(all_investor_ids)

            all_table1_list, all_table2_list, all_table3_list = [], [], []
            for item in results:
                try:
                    table1, table2, table3, verified_userno, sample_count = item
                except Exception:
                    continue
                table1.insert(0, 'userno', int(verified_userno))
                table1.insert(1, 'sample_count', sample_count)

                # 在 table2 前添加 userno 和 sample_count 列
                table2.insert(0, 'userno', int(verified_userno))
                table2.insert(1, 'sample_count', sample_count)

                # 在 table3 前添加 userno 和 sample_count 列
                table3.insert(0, 'userno', int(verified_userno))
                table3.insert(1, 'sample_count', sample_count)

                # 将每次生成的表格添加到对应的列表中
                all_table1_list.append(table1)
                all_table2_list.append(table2)
                all_table3_list.append(table3)

            all_table1 = pd.concat(all_table1_list, ignore_index=True) if all_table1_list else pd.DataFrame()
            all_table2 = pd.concat(all_table2_list, ignore_index=True) if all_table2_list else pd.DataFrame()
            all_table3 = pd.concat(all_table3_list, ignore_index=True) if all_table3_list else pd.DataFrame()

            all_table1.to_csv(os.path.join(self.output_path, "all_similar_investors.csv"), index=False)
            all_table2.to_csv(os.path.join(self.output_path, "all_recommended_products.csv"), index=False)
            all_table3.to_csv(os.path.join(self.output_path, "all_feature_contributions.csv"), index=False)

            metrics = {k: float(f"{v:.4f}") for k, v in metrics.items()}
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
            metrics_df.to_csv(os.path.join(self.output_path, "model_metrics.csv"), index=False)

            print("MABTSLR 结果保存完成")
            return self.output_path, None

    def run_cmab(self, filter_conditions, data_path, constraint_config, metrics_to_calculate):
        print("******************************")
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"算法开始运行:{now}")
        if metrics_to_calculate is None:
            metrics_to_calculate = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        # 获取当前约束配置
        current_constraint = next(iter(constraint_config.keys()))
        config = constraint_config[current_constraint]
        self.full_df = None
        # 数据预处理
        df = self.load_and_preprocess(data_path, filter_conditions)
        print(f"总用户数: {len(df['userno'].unique())}")

        # 初始化结果存储（仅当前约束）
        constraint_results = {
            current_constraint: {
                'recommendations': [],
                'similar_investors': [],
                'features': []
            }
        }
        y_true = []  # 真实标签
        y_pred = []  # 预测分数


        users = df['userno'].unique()
        results = []
        # 使用线程池，避免进程序列化开销/设备上下文问题
        with ThreadPoolExecutor(max_workers=max(1, os.cpu_count() - 1)) as executor:
            futures = [
                executor.submit(
                    self.process_user, userno, df, config, current_constraint,
                    self.m, self.T, self.alpha, self.gamma, self.lambd, self.delta, self.init_theta,
                    self.product_features
                )
                for userno in users
            ]
            for f in as_completed(futures):
                res = f.result()
                if res is not None:
                    results.append(res)
        if not results:
            return None, "没有生成任何推荐结果"
        for df, yt, yp in results:
            constraint_results[current_constraint]['recommendations'].append(df)
            y_true.extend(yt)
            y_pred.extend(yp)

        # 合并推荐结果
        rec_df = pd.concat(constraint_results[current_constraint]['recommendations'])
        rec_df.to_csv(os.path.join(self.output_path, f"all_recommended_products.csv"), index=False)

        # 评估指标计算（保持原有逻辑）
        metrics = {}
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        valid_samples = len(y_true_np) > 0
        has_positive = np.any(y_true_np == 1)
        has_negative = np.any(y_true_np == 0)

        if 'accuracy' in metrics_to_calculate and valid_samples:
            metrics['accuracy'] = accuracy_score(y_true_np, np.round(y_pred_np))
        if 'precision' in metrics_to_calculate and valid_samples:
            metrics['precision'] = precision_score(y_true_np, np.round(y_pred_np), zero_division=0)
        if 'recall' in metrics_to_calculate and valid_samples and has_positive:
            metrics['recall'] = recall_score(y_true_np, np.round(y_pred_np))
        if 'f1' in metrics_to_calculate and valid_samples and has_positive:
            metrics['f1'] = f1_score(y_true_np, np.round(y_pred_np))
        if 'auc' in metrics_to_calculate and valid_samples and has_positive and has_negative:
            metrics['auc'] = roc_auc_score(y_true_np, y_pred_np)

        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        metrics_path = os.path.join(self.output_path, "model_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"算法评估指标已保存到: {metrics_path}")
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"cmab算法结束运行:{now}")
        print("******************************")
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"mabtslr算法开始运行:{now}")
        mabtslr_runner = self.MABTSLRRecommender(output_path=self.output_path,
                                                 iterations=100,
                                                 opt_method='L-BFGS-B',
                                                 opt_options={'maxiter': 1000},
                                                 investor_features=self.investor_features,
                                                 product_features=self.product_features)
        result_path, err_msg = mabtslr_runner.run_algorithm()
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"mabtslr算法结束运行:{now}")
        print("******************************")
        if err_msg is not None:
            return None, err_msg
        try:
            self.build_radar_data()
            return self.output_path, None
        except Exception as e:
            return None, "雷达数据生成文件失败"

    def process_user(self, userno, df, config, constraint, m, T, alpha, gamma, lambd, delta, init_theta,
                     product_features):
        try:
            user_df = df[df['userno'] == userno].copy()
            if user_df.empty:
                return None, None, None
            # 对于连续字段（apr_percent, term 等），使用全局预先计算的 *_group 列进行筛选（全局四分位）
            if constraint in ['apr_percent', 'term', 'total', 'project_invest_count']:
                group_col = f"{constraint}_group"
                # 配置中的 category 采用 1..4，直接比较
                wanted = config.get('category', [])
                if group_col in user_df.columns:
                    constraint_mask = user_df[group_col].isin(wanted)
                else:
                    # 若全局分箱未生成，则降级为不过滤（或可改为返回空）
                    constraint_mask = np.ones(len(user_df), dtype=bool)
            elif constraint in ['REPAYMENT', 'level']:
                user_df[constraint] = user_df[constraint].astype(int)
                constraint_mask = user_df[constraint].isin(config['category'])
            else:
                constraint_mask = np.ones(len(user_df), dtype=bool)

            user_df = user_df[constraint_mask].copy()
            if user_df.empty or len(user_df) < m:
                return None, None, None

            X_features = user_df[product_features].values
            robo = self.RoboAdvisor(
                alpha=alpha, d=X_features.shape[1], K=X_features.shape[0], m=m,
                T=T, gamma=gamma, lambd=lambd, delta=delta, theta=init_theta, device=self.device
            )
            constraint_feature = np.ones(len(user_df), dtype=int)
            selected, _ = robo.recommend(X_features, constraint_feature)
            selected = selected.ravel()[:5]
            recommended_projects = user_df.iloc[selected]['PROJECTNO'].tolist()
            # 收集预测数据（保留核心数据收集）
            y_true = user_df['reward'].values
            valid_indices = np.arange(len(user_df))
            # q 是 torch 张量，取 CPU numpy
            y_pred = robo.q.detach().cpu().numpy()[valid_indices]
            result_df = pd.DataFrame({
                'userno': [userno] * len(recommended_projects),
                'sample_count': [len(user_df)] * len(recommended_projects),
                'Projectno': recommended_projects
            })
            return result_df, y_true, y_pred
        except Exception:
            return None, None, None

    def build_radar_data(self):
        """
        计算雷达图数据并生成对应的数据文件
        """
        try:
            fields = ['level', 'total', 'apr_percent', 'term', 'REPAYMENT', 'project_invest_count']
            project_field = "PROJECTNO"
            # 读取 筛选完的全部数据
            df = pd.read_csv(sys.get_file_path(self.output_path, "processed_data.csv"))
            # 读取 cmab算法的推荐结果
            df_cmab_recommend = pd.read_csv(sys.get_file_path(self.output_path, "all_recommended_products.csv"))
            df_cmab_recommend = df_cmab_recommend.rename(columns={"Projectno": project_field})

            # 读取 mabtslr算法的推荐结果
            df_mabtslr_recommend = pd.read_csv(
                sys.get_file_path(self.output_path, "mabtslr", "all_recommended_products.csv"))
            df_mabtslr_recommend = df_mabtslr_recommend.rename(columns={"Projectno": project_field})
            df_mabtslr_recommend[project_field] = df_mabtslr_recommend[project_field].astype(str)
            mabtslr_total = len(df_mabtslr_recommend)

            radar_field_dict = {}
            for field in fields:
                new_key = field + "_group_radar"
                # 按 field 排序
                group_num = 4
                if field == 'REPAYMENT':
                    group_num = 3
                elif field == 'level':
                    group_num = 8

                # 若是分类字段 level 或 REPAYMENT，直接使用原始数值（按 PROJECTNO 聚合取众数）
                if field in ['level', 'REPAYMENT']:
                    df_tmp = df[[project_field, field]].dropna().copy()
                    # 强制整型并按 PROJECTNO 聚合：优先取众数，否则取中位数
                    df_tmp[field] = df_tmp[field].astype(int)
                    agg = df_tmp.groupby(project_field)[field].agg(
                        lambda s: int(s.mode().iloc[0]) if not s.mode().empty else int(round(s.median()))
                    ).reset_index()
                    project_level_dict = dict(zip(agg[project_field].astype(str), agg[field].astype(int)))
                    # 针对 REPAYMENT 做 +1 偏移（0->1, 1->2, 2->3），避免雷达统计丢失 0 类
                    if field == 'REPAYMENT':
                        project_level_dict = {k: v + 1 for k, v in project_level_dict.items()}
                else:
                    # 优先使用全局预先计算的分组列（load_and_preprocess 中生成的 *_group）
                    group_col_global = f"{field}_group"
                    if group_col_global in df.columns:
                        # 聚合每个 PROJECTNO 的 group：优先取众数（mode），没有则取中位数
                        df_tmp = df[[project_field, group_col_global]].dropna().copy()
                        df_tmp[group_col_global] = df_tmp[group_col_global].astype(int)
                        agg = df_tmp.groupby(project_field)[group_col_global].agg(
                            lambda s: int(s.mode().iloc[0]) if not s.mode().empty else int(round(s.median()))
                        ).reset_index()
                        project_level_dict = dict(zip(agg[project_field].astype(str), agg[group_col_global].astype(int)))
                    else:
                        # 回退：按排序等分（用于未预计算的字段）
                        df_sorted = self.assign_equal_parts(df, field, new_key, group_num)
                        project_level_dict = dict(zip(df_sorted[project_field].astype(str), df_sorted[new_key].astype(int)))

                # 生成雷达数据（cmab 与 mabtslr）
                radar_data = {}
                radar_data['cmab'], cmab_max = self.build_radar_data_by_recommend_df(
                    df_cmab_recommend, project_level_dict, project_field, new_key, group_num=group_num
                )

                radar_data['mabtslr'], mabtslr_max = self.build_radar_data_by_recommend_df(
                    df_mabtslr_recommend, project_level_dict, project_field, new_key, group_num=group_num
                )

                mx = cmab_max
                if mabtslr_max > mx:
                    mx = mabtslr_max
                radar_data['total'] = {}
                for num in range(group_num):
                    radar_data['total'][str(num + 1)] = math.ceil(1.1 * mx)
                radar_field_dict[field] = radar_data

            with open(self.output_path + '/radar.json', 'w', encoding='utf-8') as f:
                json.dump(radar_field_dict, f, ensure_ascii=False)
        except Exception as e:
            print(e)
            print("雷达数据生成文件失败")

    @staticmethod
    def build_radar_data_by_recommend_df(recommend_df, project_level_dict, project_field, new_key, group_num: int = 4):
        '''
        根据推荐产品列表df，产品分组编号列表 计算推荐产品列表的雷达数据
        :param recommend_df:
        :param project_level_dict:
        :param project_field:
        :param new_key:
        :return:
        '''
        # 统一类型：把推荐表的 PROJECTNO 转为字符串再映射（project_level_dict 的 key 也为 str）
        recommend_df[project_field] = recommend_df[project_field].astype(str)
        recommend_df[new_key] = recommend_df[project_field].map(project_level_dict)
        radar_dict = recommend_df.groupby(new_key)[project_field].count().to_dict()
        radar_keys = list(range(1, group_num + 1))
        # 分组不足4个的 自动补零
        new_dict = {}
        max = 0
        for key in radar_keys:
            if key not in radar_dict:
                new_dict[key] = 0
            else:
                new_dict[key] = radar_dict[key]
                if radar_dict[key] > max:
                    max = radar_dict[key]
        return new_dict, max

    @staticmethod
    def assign_equal_parts(df, key: str, sort_key: str, n=4):
        """
        按从小到大排序，直接将数据等分成 n 份
        忽略边界值是否相等
        """
        # 排序
        df_sorted = df.sort_values(by=key).reset_index(drop=True).copy()

        # 每份的大小（向上取整）
        part_size = math.ceil(len(df_sorted) / n)

        # 直接用行号分组
        df_sorted[sort_key] = (df_sorted.index // part_size) + 1

        # 超过 n 的部分修正为 n
        df_sorted.loc[df_sorted[sort_key] > n, sort_key] = n

        return df_sorted


def main():
    parser = argparse.ArgumentParser(description='运行CMAB推荐算法')
    parser.add_argument('--metrics', nargs='+',
                        choices=['accuracy', 'precision', 'recall', 'f1', 'auc'],
                        default=['accuracy', 'precision', 'recall', 'f1', 'auc'],
                        help='选择评估指标，默认输出所有指标')
    args = parser.parse_args()

    # 明确定义约束配置（EHGNN风格）
    constraint_config = {
        'apr_percent': {'category': [1, 2]},  # 连续特征（利率）用分位数约束，四等分，可选填1、2、3、4
        # 'term': {'category': [1, 2]} #连续特征（借款期限）用分位数约束，四等分，可选填1、2、3、4
        # 'REPAYMENT': {'category': [0, 1]} #分类特征（还款方式）直接约束，可选填0、1、2
        # 'level': {'category': [1, 8]}  # 分类特征（风险等级）直接约束，可选填1、2、3、4、5、6、7、8
    }

    # 明确定义筛选条件（EHGNN风格）
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

    # 明确定义特征体系
    custom_investor_features = [
        'gender', 'age', 'edu_level', 'University', 'CURRENTIDENTITY',
        'credit', 'weightedbidrate_percent', 'baddebts_percent', 'user_invest_count'
    ]
    custom_product_features = [
        'total', 'apr_percent', 'term', 'REPAYMENT', 'level', 'project_invest_count'
    ]

    # 修改CMABRunner初始化（添加约束特征到输出路径）
    constraint_feature = next(iter(constraint_config.keys()))
    config_values = list(constraint_config[constraint_feature].values())[0]

    # 直接定义算法参数（EHGNN风格）
    runner = CMABRunner(
        investor_features=custom_investor_features,
        product_features=custom_product_features,
        user_id=1001,  # 直接指定用户ID
        # 新增约束特征到输出路径
        output_suffix=f"{constraint_feature}_{'_'.join(map(str, config_values))}",

        # 算法超参数配置
        alpha=0.7,  # 探索系数，取值范围(0,1]
        gamma=0.0009,  # q的步长，取值范围(0,0.01]
        lambd=0.0009,  # Q的步长，取值范围(0,0.01]
        delta=0.0009,  # theta的步长，取值范围(0,0.01]
        m=5,  # 每轮推荐数，取值范围[1,50]
        T=1000,  # 总训练轮数，取值范围[100,10000]
        init_theta=20.0  # theta初始值，取值范围[1,100]
    )
    data_path = sys.get_file_path('data/dataset/default/')
    # 运行算法
    result_path = runner.run_cmab(
        filter_conditions,
        data_path=data_path,
        constraint_config=constraint_config,
        metrics_to_calculate=args.metrics
    )

    print(f"结果保存至: {result_path}")


def load_and_prepare(file):
    """
    计算雷达图数据并生成对应的数据文件
    """
    # 读取数据
    df = pd.read_csv(sys.get_file_path(file))

    # 按 level 排序
    df_sorted = df.sort_values(by="level").reset_index(drop=True)

    # 使用 qcut 将 level 分为 4 等分，并打上标签 1,2,3,4
    df_sorted["level_group"] = pd.qcut(df_sorted["level"], q=4, labels=[1, 2, 3, 4])

    # 构建 PROJECTNO -> 等级 的映射字典
    project_level_dict = dict(zip(df_sorted["PROJECTNO"], df_sorted["level_group"]))
    count = df_sorted.groupby("level_group")["PROJECTNO"].count().to_dict()


if __name__ == "__main__":
    main()
