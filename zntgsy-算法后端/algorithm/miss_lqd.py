import os
import time
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from core import sys


# 神经网络模型定义
class CatBoostNN(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, inv_feats, pro_feats):
        return self.net(torch.cat([inv_feats, pro_feats], 1))


class NCF(nn.Module):
    def __init__(self, num_users, num_items, hidden):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, hidden)
        self.item_embed = nn.Embedding(num_items, hidden)
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embed(user_ids)
        item_emb = self.item_embed(item_ids)
        combined = torch.cat([user_emb, item_emb], 1)
        return self.predictor(combined)


class MissRecommender:
    def __init__(self, investor_features=None, product_features=None, user_id=None):
        self.full_df = None
        default_investor_features = [
            'gender', 'age', 'edu_level', 'University', 'CURRENTIDENTITY',
            'credit', 'weightedbidrate_percent',
            'baddebts_percent', 'user_invest_count'
        ]
        default_product_features = [
            'total', 'apr_percent', 'term', 'REPAYMENT', 'level',
            'project_invest_count'
        ]
        self.investor_features = investor_features if investor_features is not None else default_investor_features
        self.product_features = product_features if product_features is not None else default_product_features
        self.id_to_userno = None
        self.userno_to_id = None
        # self.output_path = os.path.join("result", "algorithm_comparison", str(int(time.time())))
        # if not os.path.exists(self.output_path):
        #     os.makedirs(self.output_path)

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

    def train_model(self):
        """Miss算法训练流程"""
        if self.full_df is None:
            print("请先加载并预处理数据。")
            return None
        print("Miss算法训练完成")
        return None

    def recommend_products(self):
        """基于余弦相似度推荐产品"""
        if self.full_df is None:
            print("请先加载并预处理数据。")
            return None

        all_investor_ids = self.full_df['investor_id'].unique()
        recommendations = []

        for investor_id in tqdm(all_investor_ids, desc="Miss算法推荐"):
            target_userno = self.id_to_userno[investor_id]
            target_data = self.full_df[self.full_df['investor_id'] == investor_id]

            # 获取所有产品
            all_products = self.full_df['product_id'].unique()

            # 基于余弦相似度的推荐逻辑
            investor_matrix = self.full_df.groupby('investor_id')[self.investor_features].mean()
            similarity = cosine_similarity(investor_matrix)

            # 找到最相似的投资者
            similar_investors = np.argsort(-similarity[investor_id])[1:6]  # 排除自己

            # 获取相似投资者购买的产品
            similar_products = []
            for sim_investor in similar_investors:
                purchased_products = self.full_df[
                    (self.full_df['investor_id'] == sim_investor) &
                    (self.full_df['reward'] == 1)
                    ]['product_id'].unique()
                similar_products.extend(purchased_products)

            # 过滤已购买产品
            purchased = self.full_df[
                (self.full_df['investor_id'] == investor_id) &
                (self.full_df['reward'] == 1)
                ]['product_id'].unique()

            # 推荐相似投资者购买但目标投资者未购买的产品
            recommended_products = list(set(similar_products) - set(purchased))

            # 如果没有推荐产品，使用基于产品特征的推荐
            if not recommended_products:
                product_matrix = self.full_df.groupby('product_id')[self.product_features].mean()
                product_similarity = cosine_similarity(product_matrix)

                # 获取目标投资者购买的产品
                target_purchased = self.full_df[
                    (self.full_df['investor_id'] == investor_id) &
                    (self.full_df['reward'] == 1)
                    ]['product_id'].unique()

                if len(target_purchased) > 0:
                    # 基于已购买产品的相似度推荐
                    for purchased_product in target_purchased:
                        similar_to_purchased = np.argsort(-product_similarity[purchased_product])[1:6]
                        recommended_products.extend(similar_to_purchased)

                recommended_products = list(set(recommended_products) - set(purchased))

            # 取前5个推荐产品
            top_products = recommended_products[:5]

            # 如果推荐产品不足5个，补充随机产品
            if len(top_products) < 5:
                all_available = list(set(all_products) - set(purchased) - set(top_products))
                additional = np.random.choice(all_available, min(5 - len(top_products), len(all_available)),
                                              replace=False)
                top_products.extend(additional)

            # 保存推荐结果
            for product_id in top_products:
                product_no = self.full_df[self.full_df['product_id'] == product_id]['PROJECTNO'].iloc[0]
                recommendations.append({
                    'userno': target_userno,
                    'Projectno': product_no
                })

        return pd.DataFrame(recommendations)


class AlgorithmComparison:
    def __init__(self,
                 user_id: str,
                 custom_investor_features: list = None,
                 custom_product_features: list = None):
        self.investor_features = custom_investor_features
        self.product_features = custom_product_features
        self.user_id = user_id
        # 添加output_path属性
        self.output_path = sys.get_file_path(sys.BASE_TASK_PATH,
                                             str(sys.ALGORITHM_TYPE_NEW_INVESTOR),
                                             str(user_id) + '_' + sys.ALGORITHM_NEW_INVESTOR_COLD_RUN_MISS,
                                             str(int(time.time())))
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def calculate_weighted_hit_rate(self, recommendations, full_df):
        """计算加权命中率（返回缩放前后结果和缩放因子）"""
        if recommendations is None or len(recommendations) == 0:
            return {'original_score': 0.0, 'scaled_score': 0.0, 'scale_factor': 1.0}

        # 创建测试集（随机抽取20%作为测试集）
        test_set = full_df.sample(frac=0.2, random_state=42)

        original_weighted_score = 0
        scaled_weighted_score = 0
        user_count = 0

        # 按用户分组推荐结果
        user_recommendations = recommendations.groupby('userno')['Projectno'].apply(list).to_dict()

        # 计算每个用户的推荐列表长度（用于缩放）
        user_recommendation_lengths = {userno: len(recommendations) for userno, recommendations in
                                       user_recommendations.items()}

        # 计算最大推荐列表长度（用于归一化缩放）
        max_recommendation_length = max(user_recommendation_lengths.values()) if user_recommendation_lengths else 1

        for userno, top5 in user_recommendations.items():
            # 获取该用户在测试集中的实际购买产品
            actual_purchases = test_set[test_set['userno'] == userno]['PROJECTNO'].tolist()

            if len(actual_purchases) == 0:
                continue  # 跳过没有实际购买记录的用户

            # 计算缩放因子：基于推荐列表长度进行归一化
            scale_factor = len(top5) / max_recommendation_length

            # 计算原始加权命中率
            user_original_score = 0
            for rank, product in enumerate(top5, 1):
                if product in actual_purchases:
                    user_original_score += 1 / rank

            # 计算缩放后的加权命中率（直接乘以缩放因子）
            user_scaled_score = user_original_score * scale_factor

            original_weighted_score += user_original_score
            scaled_weighted_score += user_scaled_score
            user_count += 1

        # 计算平均加权命中率
        if user_count > 0:
            original_avg_score = original_weighted_score / user_count
            scaled_avg_score = scaled_weighted_score / user_count

            # 移除sigmoid缩放，直接使用缩放后的结果
            # 当缩放因子为1时，缩放后命中率应该等于原始命中率

            return {
                'original_score': original_avg_score,
                'scaled_score': scaled_avg_score,
                'scale_factor': scaled_avg_score / original_avg_score if original_avg_score > 0 else 1.0
            }
        else:
            return {'original_score': 0.0, 'scaled_score': 0.0, 'scale_factor': 1.0}

    def _train_neural_model(self, model, inv_ids, pro_ids, inv_feats, pro_feats, labels, epochs=50, lr=0.001):
        """训练神经网络模型"""
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # 转换为tensor
        inv_ids_tensor = torch.LongTensor(inv_ids)
        pro_ids_tensor = torch.LongTensor(pro_ids)
        inv_feats_tensor = torch.FloatTensor(inv_feats)
        pro_feats_tensor = torch.FloatTensor(pro_feats)
        labels_tensor = torch.FloatTensor(labels)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            if isinstance(model, CatBoostNN):
                outputs = model(inv_feats_tensor, pro_feats_tensor).squeeze()
            elif isinstance(model, NCF):
                outputs = model(inv_ids_tensor, pro_ids_tensor).squeeze()

            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            if isinstance(model, CatBoostNN):
                preds = model(inv_feats_tensor, pro_feats_tensor).squeeze()
            elif isinstance(model, NCF):
                preds = model(inv_ids_tensor, pro_ids_tensor).squeeze()

        return preds

    def _get_common_data(self, full_df):
        """获取训练数据"""
        inv_ids = full_df['investor_id'].values
        pro_ids = full_df['product_id'].values
        inv_feats = full_df[self.investor_features].values
        pro_feats = full_df[self.product_features].values
        labels = full_df['reward'].values
        return inv_ids, pro_ids, inv_feats, pro_feats, labels

    def _run_cf_model(self, full_df, cf_type):
        """运行协同过滤模型"""
        interaction = pd.pivot_table(full_df, values='reward',
                                     index='investor_id', columns='product_id',
                                     fill_value=0)
        if cf_type == 'investor':
            sim = cosine_similarity(interaction)
            preds = np.dot(sim, interaction.values) / (np.sum(np.abs(sim), axis=1) + 1e-9)[:, None]
        else:  # product
            sim = cosine_similarity(interaction.T)
            preds = np.dot(interaction.values, sim) / (np.sum(np.abs(sim), axis=1) + 1e-9)
        return (preds.flatten() > 0.5).astype(int)

    def run_catboostnn(self, full_df):
        """运行真正的CatBoostNN算法"""
        print("运行CatBoostNN算法...")

        # 获取训练数据
        inv_ids, pro_ids, inv_feats, pro_feats, labels = self._get_common_data(full_df)

        # 创建模型
        model = CatBoostNN(
            input_dim=len(self.investor_features) + len(self.product_features),
            hidden=64
        )

        # 训练模型
        pred_scores = self._train_neural_model(model, inv_ids, pro_ids, inv_feats, pro_feats, labels)
        pred_scores = pred_scores.detach().numpy()

        # 生成推荐结果
        recommendations = []
        for investor_id in full_df['investor_id'].unique():
            investor_mask = (full_df['investor_id'] == investor_id)
            investor_scores = pred_scores[investor_mask]
            sorted_indices = np.argsort(-investor_scores)[:5]
            recommended_projects = full_df[investor_mask].iloc[sorted_indices]['product_id'].tolist()

            # 转换为原始ID
            for product_id in recommended_projects:
                product_no = full_df[full_df['product_id'] == product_id]['PROJECTNO'].iloc[0]
                userno = full_df[full_df['investor_id'] == investor_id]['userno'].iloc[0]
                recommendations.append({'userno': userno, 'Projectno': product_no})

        return pd.DataFrame(recommendations)

    def run_investor_cf(self, full_df):
        """运行基于投资者的协同过滤算法"""
        print("运行基于投资者的协同过滤算法...")

        # 使用向量化计算
        preds = self._run_cf_model(full_df, 'investor')

        recommendations = []
        for investor_id in full_df['investor_id'].unique():
            investor_mask = (full_df['investor_id'] == investor_id)
            investor_scores = preds[investor_mask]
            sorted_indices = np.argsort(-investor_scores)[:5]
            recommended_projects = full_df[investor_mask].iloc[sorted_indices]['product_id'].tolist()

            # 转换为原始ID
            for product_id in recommended_projects:
                product_no = full_df[full_df['product_id'] == product_id]['PROJECTNO'].iloc[0]
                userno = full_df[full_df['investor_id'] == investor_id]['userno'].iloc[0]
                recommendations.append({'userno': userno, 'Projectno': product_no})

        return pd.DataFrame(recommendations)

    def run_product_cf(self, full_df):
        """运行基于产品的协同过滤算法"""
        print("运行基于产品的协同过滤算法...")

        # 使用向量化计算
        preds = self._run_cf_model(full_df, 'product')

        recommendations = []
        for investor_id in full_df['investor_id'].unique():
            investor_mask = (full_df['investor_id'] == investor_id)
            investor_scores = preds[investor_mask]
            sorted_indices = np.argsort(-investor_scores)[:5]
            recommended_projects = full_df[investor_mask].iloc[sorted_indices]['product_id'].tolist()

            # 转换为原始ID
            for product_id in recommended_projects:
                product_no = full_df[full_df['product_id'] == product_id]['PROJECTNO'].iloc[0]
                userno = full_df[full_df['investor_id'] == investor_id]['userno'].iloc[0]
                recommendations.append({'userno': userno, 'Projectno': product_no})

        return pd.DataFrame(recommendations)

    def run_ncf(self, full_df):
        """运行神经协同过滤算法"""
        print("运行神经协同过滤算法...")

        # 获取训练数据
        inv_ids, pro_ids, inv_feats, pro_feats, labels = self._get_common_data(full_df)

        # 创建模型
        model = NCF(
            num_users=full_df['investor_id'].nunique(),
            num_items=full_df['product_id'].nunique(),
            hidden=64
        )

        # 训练模型
        pred_scores = self._train_neural_model(model, inv_ids, pro_ids, inv_feats, pro_feats, labels)
        pred_scores = pred_scores.detach().numpy()

        # 生成推荐结果
        recommendations = []
        for investor_id in full_df['investor_id'].unique():
            investor_mask = (full_df['investor_id'] == investor_id)
            investor_scores = pred_scores[investor_mask]
            sorted_indices = np.argsort(-investor_scores)[:5]
            recommended_projects = full_df[investor_mask].iloc[sorted_indices]['product_id'].tolist()

            # 转换为原始ID
            for product_id in recommended_projects:
                product_no = full_df[full_df['product_id'] == product_id]['PROJECTNO'].iloc[0]
                userno = full_df[full_df['investor_id'] == investor_id]['userno'].iloc[0]
                recommendations.append({'userno': userno, 'Projectno': product_no})

        return pd.DataFrame(recommendations)

    def run_comparison(self, filter_conditions, data_path="E:/桌面/研一下/导师任务/智能投顾平台/Data"):
        """运行算法对比"""
        print("开始算法对比...")

        # 1. 数据预处理
        miss_recommender = MissRecommender(investor_features=self.investor_features,
                                           product_features=self.product_features)
        full_df = miss_recommender.load_and_preprocess_data(data_path, filter_conditions)
        if full_df is None:
            print("数据加载失败")
            return None, "数据加载失败"

        # # 设置特征列表用于其他算法
        # self.investor_features = miss_recommender.investor_features
        # self.product_features = miss_recommender.product_features

        # 2. 运行各算法
        algorithms = {
            'miss': miss_recommender.recommend_products,
            'catboostnn': lambda: self.run_catboostnn(full_df),
            'investor_cf': lambda: self.run_investor_cf(full_df),
            'product_cf': lambda: self.run_product_cf(full_df),
            'ncf': lambda: self.run_ncf(full_df)
        }

        results = {}

        for algo_name, algo_func in algorithms.items():
            print(f"\n运行{algo_name}算法...")
            try:
                recommendations = algo_func()
                results[algo_name] = recommendations

                # 保存推荐结果
                output_file = os.path.join(self.output_path, f"{algo_name}_recommendations.csv")
                recommendations.to_csv(output_file, index=False)
                print(f"{algo_name}推荐结果已保存到: {output_file}")

            except Exception as e:
                print(f"{algo_name}算法运行失败: {e}")
                results[algo_name] = None

        # 3. 计算加权命中率（包含缩放前后结果）
        comparison_results = []

        for algo_name, recommendations in results.items():
            if recommendations is not None:
                hit_rate_info = self.calculate_weighted_hit_rate(recommendations, full_df)

                # 只包含四个列，按指定顺序：Algorithm、Weighted_HitRate、Original_HitRate、Scale_Factor
                comparison_results.append({
                    'Algorithm': algo_name,
                    'Weighted_HitRate': hit_rate_info['scaled_score'],  # 缩放后命中率
                    'Original_HitRate': hit_rate_info['original_score'],
                    'Scale_Factor': hit_rate_info['scale_factor']
                })

                print(f"{algo_name}加权命中率:")
                print(f"  原始命中率: {hit_rate_info['original_score']:.6f}")
                print(f"  缩放后命中率: {hit_rate_info['scaled_score']:.6f}")
                print(f"  缩放因子: {hit_rate_info['scale_factor']:.6f}")

        # 4. 保存对比结果
        comparison_df = pd.DataFrame(comparison_results)
        comparison_file = os.path.join(self.output_path, "comparison_results.csv")
        comparison_df.to_csv(comparison_file, index=False)
        print(f"\n算法对比结果已保存到: {comparison_file}")

        # 5. 打印最终结果
        print("\n最终算法对比结果:")
        print(comparison_df.to_string(index=False, float_format='%.6f'))

        # 6. 生成详细报告
        self._generate_detailed_report(comparison_df, results, full_df)
        return self.output_path, None

    def _generate_detailed_report(self, comparison_df, results, full_df):
        """生成详细对比报告"""
        report_file = os.path.join(self.output_path, "detailed_comparison_report.txt")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("算法对比详细报告\n")
            f.write("=" * 50 + "\n\n")

            f.write("1. 算法性能对比:\n")
            f.write("-" * 30 + "\n")
            for _, row in comparison_df.iterrows():
                f.write(f"算法: {row['Algorithm']}\n")
                f.write(f"  加权命中率: {row['Weighted_HitRate']:.6f}\n")
                f.write(f"  原始命中率: {row['Original_HitRate']:.6f}\n")
                f.write(f"  缩放因子: {row['Scale_Factor']:.6f}\n")
                f.write("\n")

            f.write("2. 推荐结果统计:\n")
            f.write("-" * 30 + "\n")
            for algo_name, recommendations in results.items():
                if recommendations is not None:
                    f.write(f"算法 {algo_name}:\n")
                    f.write(f"  推荐用户数: {recommendations['userno'].nunique()}\n")
                    f.write(f"  推荐产品数: {recommendations['Projectno'].nunique()}\n")
                    f.write(f"  总推荐数: {len(recommendations)}\n")
                    f.write("\n")

            f.write("3. 数据统计:\n")
            f.write("-" * 30 + "\n")
            f.write(f"总用户数: {full_df['userno'].nunique()}\n")
            f.write(f"总产品数: {full_df['PROJECTNO'].nunique()}\n")
            f.write(f"总交互记录: {len(full_df)}\n")

        print(f"详细对比报告已保存到: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='运行Miss算法与四个传统算法的对比')

    args = parser.parse_args()

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

    # 运行算法对比
    comparison = AlgorithmComparison(user_id="default_user",
                                     custom_product_features=custom_product_features,
                                     custom_investor_features=custom_investor_features)
    comparison.run_comparison(filter_conditions, data_path="c:/code/zntgsy/data/dataset/default")


if __name__ == '__main__':
    main()
