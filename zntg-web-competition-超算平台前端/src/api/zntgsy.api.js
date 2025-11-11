import request from "@/utils/request";

const AUTH_BASE_URL = "/algorithm-api";

const ZntgsyAPI = {
  /** 执行算法任务*/
  postRunTask(data) {
    return request({
      url: `${AUTH_BASE_URL}/run-task`,
      method: "post",
      data: data,
    });
  },
  /** 获取实验数据集列表*/
  getDataList() {
    return request({
      url: `${AUTH_BASE_URL}/ex-data/list`,
      method: "get",
    });
  },
  /** 获取相关网络数据*/
  postNetwork(data) {
    return request({
      url: `${AUTH_BASE_URL}/query-network`,
      method: "post",
      data: data,
    });
  },
  /** 获取实验对象属性*/
  getQueryFeatures(params) {
    return request({
      url: `${AUTH_BASE_URL}/ex-data/query-features`,
      method: "get",
      params: params,
    });
  },
  /** 获取已有算法*/
  getAlgorithmList(params) {
    return request({
      url: `${AUTH_BASE_URL}/algorithm-list`,
      method: "get",
      params: params,
    });
  },
  /** 获取已有投资者列表*/
  getInvestors(params) {
    return request({
      url: `${AUTH_BASE_URL}/ex-data/investors`,
      method: "get",
      params: params,
    });
  },
  /** 获取投资者详情*/
  getInvestorsDetail(params) {
    return request({
      url: `${AUTH_BASE_URL}/ex-data/investor-detail`,
      method: "get",
      params: params,
    });
  },
  /** 获取算法评价*/
  getTaskMetrics(params) {
    return request({
      url: `${AUTH_BASE_URL}/ex-data/query-task-metrics`,
      method: "get",
      params: params,
    });
  },
  /** 获取入参参数*/
  getTaskParams(params) {
    return request({
      url: `${AUTH_BASE_URL}/ex-data/query-task-params`,
      method: "get",
      params: params,
    });
  },
};

export default ZntgsyAPI;
