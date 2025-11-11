import request from "@/utils/request";

const AUTH_BASE_URL = "/api";

const SyjgckAPI = {
  /** 保存实验结果 */
  postSaveExp(data) {
    return request({
      url: `${AUTH_BASE_URL}/exp-data/save-exp-result`,
      method: "post",
      data: data,
    });
  },
  /** 分页查询实验结果 */
  postPageExp(data) {
    return request({
      url: `${AUTH_BASE_URL}/exp-data/page-exp-result`,
      method: "post",
      data: data
    });
  },
};

export default SyjgckAPI;
