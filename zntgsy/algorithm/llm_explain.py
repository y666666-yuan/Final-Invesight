import os
import time

from core import sys
from algorithm.llm_gnn.data_splite import DataSplitterCache
from algorithm.llm_gnn.utils.data_handler import DataHandlerCache
from algorithm.llm_gnn.train_gcn import TrainGNNCache


class LLMExplainRunner:
    def __init__(self, investor_features=None, product_features=None, user_id=None):
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
        self.output_path = sys.get_file_path(sys.BASE_TASK_PATH,
                                             str(sys.ALGORITHM_TYPE_EXPLAIN),
                                             str(user_id) + '_' + sys.ALGORITHM_EXPLAIN_GCN,
                                             str(int(time.time())))
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def run(self, filter_conditions, data_path):
        csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
        if len(csv_files) != 1:
            print(f"错误：路径 {data_path} 下应只有一个 .csv 文件，当前找到 {len(csv_files)} 个。")
            return None
        csv_file = csv_files[0]
        splitter = DataSplitterCache(total_file=csv_file, filter_conditions=filter_conditions)
        splitter.split_data()
        data_handler_cache = DataHandlerCache(para_dict=splitter.cache_data['para_dict'],
                                              train_data=splitter.cache_data['train_data'],
                                              val_data=splitter.cache_data['val_data'],
                                              test_data=splitter.cache_data['test_data'],
                                              base_data=splitter.df_base,
                                              base_data_yuan=splitter.df_baseyuan)
        trainer = TrainGNNCache(data_handler_cache=data_handler_cache,
                                investor_features=self.investor_features,
                                product_features=self.product_features,
                                output_path=self.output_path)
        trainer.train(filter_conditions=filter_conditions, data_path=data_path)
        return self.output_path, None


if __name__ == "__main__":
    # 定义样本筛选维度
    filter_conditions = {
        "gender": None,
        "age": {
            "min": 40,
            "max": 50
        },
        "current_identity": None,
        "edu_level": None,
        "user_invest_count": {
            "min": 1,
            "max": 30
        },
        "total": {
            "min": 1000,
            "max": 100000
        },
        "apr_percent": {
            "min": 0,
            "max": 1
        },
        "repayment": None,
        "term": {
            "min": 1,
            "max": 6
        },
        "level": {
            "min": 1,
            "max": 8
        },
        "project_invest_count": {
            "min": 0,
            "max": 200
        }
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
    runner = LLMExplainRunner(investor_features=custom_investor_features,
                              product_features=custom_product_features,
                              user_id="1")
    runner.run(filter_conditions=filter_conditions, data_path=sys.get_file_path('data/dataset/default/'))
