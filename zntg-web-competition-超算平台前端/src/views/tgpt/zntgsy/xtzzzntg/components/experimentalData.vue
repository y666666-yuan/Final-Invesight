<template>
  <div class="exper" v-loading="experLoading" element-loading-text="计算中...">
    <div class="exper-main">
      <div class="exper-title">
        <div class="i-svg:title"></div>
        <div class="exper-name">实验数据选择</div>
      </div>
      <div class="exper-con">
        <div class="con-name">
          <div class="con-left">
            <p>*</p>
            <p>数据来源：</p>
          </div>
          <div class="con-list">
            <div
              v-for="(item, index) in chackArr"
              :key="index"
              :class="['list-con', intelligenceActive === index ? 'list-con_active' : '']"
              @click="handleList(index)"
            >
              <div :class="`i-svg:${item.icon}`"></div>
              <div>{{ item.name }}</div>
            </div>
          </div>
        </div>
        <div class="con-name">
          <div class="con-left">
            <p>*</p>
            <p>{{ intelligenceActive === 0 ? '数据名称：' : '上传数据：'}}</p>
          </div>
          <div class="con-select" v-if="intelligenceActive === 0">
            <el-select
              v-model="datasetObj.dataset_name"
              placeholder="请选择数据名称"
              style="width: 320px"
              @change="handleDatasetName"
            >
              <el-option v-for="item in dataNameArr" :key="item" :label="item" :value="item" />
            </el-select>
          </div>
          <el-upload
            v-if="intelligenceActive === 1"
            class="upload-demo"
            drag
            action="/sypt/algorithm-api/ex-data/upload/dataset"
            :before-upload="beforeUpload"
            :on-success="handleSuccess"
            :on-error="handleError"
            :limit="1"
            :file-list="fileList"
            accept=".csv"
            :auto-upload="true"
            :headers="uploadHeaders"
          >
            <i class="el-icon-upload"></i>
            <div class="el-upload__text">
              <em>点击上传</em>
            </div>
            <template #tip>
              <div class="el-upload__tip">
                只支持上传<i>.csv</i>文件，文件大小不超过<i>40MB</i>
              </div>
            </template>
          </el-upload>
          <a v-if="intelligenceActive === 1" class="el-down" href="/sypt/data/自行上传数据说明.xlsx">
            <div class="i-svg:download"></div>
            <p>下载数据结构说明</p>
          </a>
        </div>
        <div class="intelligence-line">
          <p></p>
          <p>投资者筛选</p>
          <p @click="closeSelect(1)">一键恢复</p>
        </div>
        <div class="intelligence-checkbox">
          <div class="checkbox-con">
            <p class="custom-checkbox">性别：</p>
            <el-select
              v-model="datasetObj.filter.gender"
              multiple
              collapse-tags
              :max-collapse-tags="3"
              placeholder="请选择"
              style="width: 320px"
              clearable
            >
              <el-option
                v-for="(item, index) in genderArr"
                :key="index"
                :label="item.label"
                :value="item.value"
              />
            </el-select>
          </div>
          <div class="checkbox-con">
            <p class="custom-checkbox">年龄：</p>
            <div class="slider-demo-block">
              <el-slider
                v-model="datasetObj.filter.age"
                range
                :min="20"
                :max="60"
                :marks="{ 20: '20', 60: '60' }"
              />
            </div>
          </div>
          <div class="checkbox-con">
            <p class="custom-checkbox">教育程度：</p>
            <el-select
              v-model="datasetObj.filter.edu_level"
              multiple
              collapse-tags
              :max-collapse-tags="2"
              placeholder="请选择"
              style="width: 320px"
              clearable
            >
              <el-option
                v-for="(item, index) in eduLevelArr"
                :key="index"
                :label="item.label"
                :value="item.value"
              />
            </el-select>
          </div>
          <div class="checkbox-con">
            <p class="custom-checkbox">身份：</p>
            <el-select
              v-model="datasetObj.filter.current_identity"
              multiple
              collapse-tags
              :max-collapse-tags="2"
              placeholder="请选择"
              style="width: 320px"
              clearable
            >
              <el-option
                v-for="(item, index) in currentIdentityArr"
                :key="index"
                :label="item.label"
                :value="item.value"
              />
            </el-select>
          </div>
          <div class="checkbox-con">
            <p class="custom-checkbox">投资次数：</p>
            <div class="slider-demo-block">
              <el-slider
                v-model="datasetObj.filter.user_invest_count"
                range
                :min="1"
                :max="30"
                :marks="{ 1: '1', 30: '30' }"
              />
            </div>
          </div>
        </div>
        <div class="intelligence-line intelligence-line_two">
          <p></p>
          <p>产品筛选</p>
          <p @click="closeSelect(2)">一键恢复</p>
        </div>
        <div class="intelligence-checkbox">
          <div class="checkbox-con">
            <p class="custom-checkbox">规模：</p>
            <div class="slider-demo-block">
              <el-slider
                v-model="datasetObj.filter.total"
                range
                :min="1000"
                :max="100000"
                :marks="{ 1000: '1000', 100000: '100000' }"
              />
            </div>
          </div>
          <div class="checkbox-con">
            <p class="custom-checkbox">利率：</p>
            <div class="slider-demo-block">
              <el-slider
                v-model="datasetObj.filter.apr_percent"
                :step="0.1"
                range
                :min="0"
                :max="1"
                :marks="{ 0: '0', 1: '1' }"
              />
            </div>
          </div>
          <div class="checkbox-con">
            <p class="custom-checkbox">还款方式：</p>
            <el-select
              v-model="datasetObj.filter.repayment"
              multiple
              collapse-tags
              :max-collapse-tags="1"
              placeholder="请选择"
              style="width: 320px"
              clearable
            >
              <el-option
                v-for="(item, index) in repaymentArr"
                :key="index"
                :label="item.label"
                :value="item.value"
              />
            </el-select>
          </div>
          <div class="checkbox-con">
            <p class="custom-checkbox">借款期限：</p>
            <div class="slider-demo-block">
              <el-slider
                v-model="datasetObj.filter.term"
                range
                :min="1"
                :max="24"
                :marks="{ 1: '1', 24: '24' }"
              />
            </div>
          </div>
          <div class="checkbox-con">
            <p class="custom-checkbox">风险等级：</p>
            <div class="slider-demo-block">
              <el-slider
                v-model="datasetObj.filter.level"
                range
                :min="1"
                :max="8"
                :marks="{ 1: '1', 8: '8' }"
              />
            </div>
          </div>
          <div class="checkbox-con">
            <p class="custom-checkbox">被投资次数：</p>
            <div class="slider-demo-block">
              <el-slider
                v-model="datasetObj.filter.project_invest_count"
                range
                :min="0"
                :max="200"
                :marks="{ 1: '0', 200: '200' }"
              />
            </div>
          </div>
        </div>
      </div>
      <div class="exper-footer">
        <!-- <div class="exper-request" @click="handleRequest()">
          <div class="i-svg:sysjksh"></div>
          <div>实验数据可视化</div>
        </div> -->
        <p class="exper-step" @click="handleStep(2)">下一步</p>
      </div>
    </div>
    <div v-if="visualizationBol" class="exper-visualization" v-loading="loading" element-loading-text="加载中...">
      <div class="visualization-title">
        <div class="i-svg:title"></div>
        <div class="visualization-name">可视化结果</div>
      </div>
      <div class="visualization-main" v-if="relationData && relationData.nodes && relationData.nodes.length > 0">
        <div class="visualization-tab">
          <el-tabs v-model="datasetObj.network_type" class="demo-tabs" @tab-change="handleClick">
            <el-tab-pane label="投资者-产品网络" :name="0"></el-tab-pane>
            <el-tab-pane label="投资者网络" :name="1"></el-tab-pane>
            <el-tab-pane label="产品网络" :name="2"></el-tab-pane>
          </el-tabs>
        </div>
        <div class="visualization-chart">
          <div class="chart-con">
            <RelationGraph
              v-if="relationData && relationData.nodes && relationData.nodes.length > 0 && networkBol"
              id="chartContainer"
              :disableScroll="true"
              :chartData="relationData"
            />
            <div v-else class="chart-none">自定义数据无该网络关系</div>
          </div>
          <div class="chart-leng">
            <div class="leng-left">
              <p></p>
              <p>投资者</p>
            </div>
            <div class="leng-right">
              <p></p>
              <p>产品</p>
            </div>
          </div>
        </div>
      </div>
      <div class="visualization-main" v-if="relationData && relationData.nodes && relationData.nodes.length === 0 && !loading">
        <p class="visualization-error">无符合条件的实验数据，请修改后重试。</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { cloneDeep } from "lodash-es";
import { ref, reactive, toRefs, onMounted, onBeforeUnmount, computed } from "vue";
import ZntgsyAPI from "@/api/zntgsy.api";
import { getAccessToken } from "@/utils/auth";
import RelationGraph from "./bar/newrelationGraph.vue";
const emits = defineEmits(["goNextStep"]);
const props = defineProps({
  data: {
    default: {},
    type: Object,
  },
});
const intelligenceActive = ref(0);
const fileList = ref([])
const state = reactive({
  visualizationBol: false,
  loading: true,
  experLoading: false,
  chackArr: [
    {
      name: "平台内嵌数据",
      icon: "pt",
    },
    {
      name: "自定义数据",
      icon: "xiugai",
    },
  ],
  dataNameArr: [],
  datasetObj:
    props.data && Object.keys(props.data).length > 0
      ? props.data
      : {
          network_type: 0,
          dataset_name: "",
          filter: {
            gender: [],
            age: [40, 50],
            current_identity: [],
            edu_level: [],
            user_invest_count: [1, 30],
            total: [1000, 100000],
            apr_percent: [0, 1],
            repayment: [],
            term: [1, 15],
            level: [1, 8],
            project_invest_count: [0, 200],
          },
        },
  genderArr: [
    {
      label: "男",
      value: 0,
    },
    {
      label: "女",
      value: 1,
    },
  ],
  eduLevelArr: [
    {
      label: "初中以下",
      value: 0,
    },
    {
      label: "高中",
      value: 1,
    },
    {
      label: "本科",
      value: 2,
    },
    {
      label: "研究生及以上",
      value: 3,
    },
  ],
  currentIdentityArr: [
    {
      label: "工薪族",
      value: 1,
    },
    {
      label: "私营业主",
      value: 2,
    },
    {
      label: "其他",
      value: 3,
    },
    {
      label: "网店卖家",
      value: 4,
    },
    {
      label: "学生",
      value: 5,
    },
  ],
  repaymentArr: [
    {
      label: "等额本息",
      value: 0,
    },
    {
      label: "一次性还本付息",
      value: 1,
    },
    {
      label: "月还息，季还1/4本金",
      value: 2,
    },
  ],
  termArr: [
    {
      label: "1个月",
      value: 1,
    },
    {
      label: "3个月",
      value: 3,
    },
    {
      label: "6个月",
      value: 6,
    },
    {
      label: "7个月",
      value: 7,
    },
    {
      label: "9个月",
      value: 9,
    },
    {
      label: "12个月",
      value: 12,
    },
    {
      label: "18个月",
      value: 18,
    },
    {
      label: "24个月",
      value: 24,
    },
  ],
  relationData: {},
  networkBol: true,
  uploadHeaders:{
    Authorization: getAccessToken(),
  }
});
const {
  visualizationBol,
  loading,
  experLoading,
  chackArr,
  dataNameArr,
  datasetObj,
  genderArr,
  eduLevelArr,
  currentIdentityArr,
  repaymentArr,
  termArr,
  relationData,
  networkBol,
  uploadHeaders,
} = toRefs(state);
onMounted(() => {
  getDataList();
});
const getDataList = () => {
  ZntgsyAPI.getDataList().then((data) => {
    state.dataNameArr = data;
  });
};
const handleDatasetName = (val) => {
  emits("handleDataset", val);
};
const handleList = (num) => {
  intelligenceActive.value = num;
};
const closeSelect = (num) => {
  // 一键恢复
  if (num === 1) {
    state.datasetObj.filter.gender = []
    state.datasetObj.filter.age = [40, 50]
    state.datasetObj.filter.current_identity = []
    state.datasetObj.filter.edu_level = []
    state.datasetObj.filter.user_invest_count = [1, 30]
  } else {
    state.datasetObj.filter.total = [1000, 100000]
    state.datasetObj.filter.apr_percent = [0, 1]
    state.datasetObj.filter.repayment = []
    state.datasetObj.filter.term = [1, 15]
    state.datasetObj.filter.level = [1, 8]
    state.datasetObj.filter.project_invest_count = [0, 200]
  }
}
const handleRequest = () => {
  // 查看可视化结果
  if (state.datasetObj.dataset_name) {
    state.relationData = {};
    postNetwork();
  } else {
    ElMessage.warning("请选择数据名称！");
  }
};
const postNetwork = (num) => {
  // 获取相关网络数据
  if (!num) {
    state.visualizationBol = true;
  } else {
    state.experLoading = true;
  }
  state.loading = true;
  let params = cloneDeep(state.datasetObj);
  params.filter.gender = params.filter.gender.length > 0 ? params.filter.gender : null;
  params.filter.edu_level = params.filter.edu_level.length > 0 ? params.filter.edu_level : null;
  params.filter.current_identity =
    params.filter.current_identity.length > 0 ? params.filter.current_identity : null;
  params.filter.repayment = params.filter.repayment.length > 0 ? params.filter.repayment : null;
  params.filter.age = { min: params.filter.age[0], max: params.filter.age[1] };
  params.filter.user_invest_count = {
    min: params.filter.user_invest_count[0],
    max: params.filter.user_invest_count[1],
  };
  params.filter.total = { min: params.filter.total[0], max: params.filter.total[1] };
  params.filter.apr_percent = {
    min: params.filter.apr_percent[0],
    max: params.filter.apr_percent[1],
  };
  params.filter.term = { min: params.filter.term[0], max: params.filter.term[1] };
  params.filter.level = { min: params.filter.level[0], max: params.filter.level[1] };
  params.filter.project_invest_count = {
    min: params.filter.project_invest_count[0],
    max: params.filter.project_invest_count[1],
  };
  ZntgsyAPI.postNetwork(params).then((data) => {
    state.relationData = data;
    if (num) {
      state.experLoading = false;
      if (state.relationData.nodes.length > 0) {
        emits("goNextStep", num, "down", state.datasetObj);
      } else {
        ElMessage.error("无符合条件的实验数据，请修改后重试");
      }
    } else {
      state.networkBol = true;
      state.loading = false;
    }
  }).catch((error) => {
    state.networkBol = false;
    state.loading = false;
  });
};
const handleClick = (val) => {
  // 可视化结果tab切换
  state.datasetObj.network_type = val;
  postNetwork();
};
const handleStep = (num) => {
  // 下一步
  if (state.datasetObj.dataset_name) {
    postNetwork(num)
  } else {
    ElMessage.warning("请选择数据名称！");
  }
};
const beforeUpload = (file) => {
  const isCSV = file.type === 'text/csv' || file.name.endsWith('.csv')
  const isLt40M = file.size / 1024 / 1024 <= 40

  if (!isCSV) {
    ElMessage.error('只能上传 CSV 格式文件！')
    return false
  }

  if (!isLt40M) {
    ElMessage.error('上传文件大小不能超过 40MB！')
    return false
  }

  // 如果已经上传了文件，替换掉
  fileList.value = [file]
  return true
}

const handleSuccess = (response, file, fileList) => {
  console.log(file)
  if (response.code === 200) {
    state.datasetObj.dataset_name = file.name.replace('.csv', '')
    ElMessage.success('上传成功！')
  } else {
    ElMessage.error(response.data)
  }
}

const handleError = (err) => {
  ElMessage.error('上传失败，请重试！')
  console.error(err)
}
</script>

<style lang="scss" scoped>
.exper {
  display: flex;
  flex-direction: column;
  width: 100%;
  background: #f6f6f6;
  .exper-main {
    width: 100%;
    // height: 698px;
    padding-bottom: 17px;
    background: #ffffff;
    box-shadow: 0px 2px 4px 0px rgba(0, 0, 0, 0.06);
    .exper-title {
      display: flex;
      align-items: center;
      width: 100%;
      height: 45px;
      padding-left: 16px;
      border-bottom: 1px solid #eeeeee;
      color: #003d67;
      .exper-name {
        height: 22px;
        margin-left: 8px;
        font-family:
          PingFangSC,
          PingFang SC;
        font-weight: 600;
        font-size: 16px;
        line-height: 22px;
      }
    }
    .exper-con {
      display: flex;
      flex-direction: column;
      width: 100%;
      .con-name {
        display: flex;
        width: 100%;
        height: auto;
        margin-top: 24px;
        padding: 0 40px;
        .con-left{
          display: flex;
          align-items: center;
          height: 36px;
        }
        p {
          margin: 0;
          padding: 0;
          font-size: 16px;
        }
        p:nth-of-type(1) {
          height: 11px;
          color: #ff0000;
          line-height: 22px;
        }
        p:nth-of-type(2) {
          height: 22px;
          margin-left: 8px;
          font-family:
            PingFangSC,
            PingFang SC;
          font-weight: 400;
          font-size: 16px;
          color: rgba(0, 0, 0, 0.85);
          line-height: 22px;
        }
        .con-list {
          display: flex;
          align-items: center;
          height: 36px;
          border-radius: 2px;
          border: 1px solid #e1e1e1;
          box-sizing: border-box;
          .list-con {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 160px;
            height: 34px;
            background: #ffffff;
            border-radius: 2px 0px 0px 2px;
            font-family:
              PingFangSC,
              PingFang SC;
            font-weight: 400;
            font-size: 16px;
            color: rgba(0, 0, 0, 0.85);
            // line-height: 34px;
            text-align: center;
            cursor: pointer;
            div:nth-of-type(1) {
              margin: 1px 9px 0 0;
            }
          }
          .list-con_active {
            background: #eef8ff;
            color: #009eff;
          }
        }
        .el-down{
          display: flex;
          justify-content: center;
          align-items: center;
          height: 36px;
          margin-left: 20px;
          p{
            margin: 0 0 0 3px;
            padding: 0;
            height: 36px;
            line-height: 36px;
            font-size: 14px;
            color: rgba(0, 0, 0, 0.65);
          }
        }
      }
      .con-name:nth-of-type(2) {
        margin-top: 32px;
      }
      .intelligence-line {
        display: flex;
        align-items: center;
        width: calc(100% - 48px);
        margin: 32px auto 0 24px;
        height: 22px;
        p {
          margin: 0;
          padding: 0;
        }
        p:nth-of-type(1) {
          width: 2px;
          height: 14px;
          margin-top: 2px;
          background: #003d67;
          border-radius: 1px;
        }
        p:nth-of-type(2) {
          height: 22px;
          margin-left: 10px;
          font-family:
            PingFangSC,
            PingFang SC;
          font-weight: 600;
          font-size: 16px;
          color: #003d67;
          line-height: 22px;
        }
        p:nth-of-type(3) {
          height: 22px;
          margin-left: auto;
          font-family:
            PingFangSC,
            PingFang SC;
          font-weight: 400;
          font-size: 16px;
          color: #ff0000;
          line-height: 22px;
          cursor: pointer;
        }
      }
      .intelligence-line_two {
        margin-top: 24px;
      }
      .intelligence-checkbox {
        display: flex;
        flex-wrap: wrap;
        width: calc(100% - 48px);
        height: 162px;
        margin: 12px 24px 0 24px;
        background-color: #f4faff;
        .checkbox-con {
          display: flex;
          align-items: center;
          width: auto;
          height: 36px;
          margin: 24px 0 8px 0;
          .custom-checkbox {
            height: 36px;
            line-height: 36px;
            margin: 0;
            padding: 0;
            text-align: right;
            font-family:
              PingFangSC,
              PingFang SC;
            font-weight: 400;
            font-size: 16px;
            color: rgba(0, 0, 0, 0.85);
          }
          .slider-demo-block {
            width: 320px;
            display: flex;
            // align-items: center;
          }
          .slider-demo-block .el-slider {
            margin-top: 5px;
            margin-left: 0;
          }
        }
        .checkbox-con:nth-of-type(3n + 1) {
          .custom-checkbox {
            width: 112px;
          }
        }
        .checkbox-con:nth-of-type(3n + 2) {
          .custom-checkbox {
            width: 216px;
          }
        }
        .checkbox-con:nth-of-type(3n + 3) {
          .custom-checkbox {
            width: 216px;
          }
        }
      }
    }
    .exper-footer {
      display: flex;
      align-items: center;
      margin: 24px auto 0 24px;
      .exper-request {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 156px;
        height: 36px;
        background: #eef8ff;
        box-shadow: 0px 2px 4px 0px rgba(0, 0, 0, 0.06);
        border-radius: 2px;
        border: 1px solid #009eff;
        font-family:
          PingFangSC,
          PingFang SC;
        font-weight: 400;
        font-size: 16px;
        color: #009eff;
        cursor: pointer;
        div:nth-of-type(2) {
          margin-left: 7px;
        }
      }
      .exper-step {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 156px;
        height: 36px;
        margin: 0;
        padding: 0;
        // margin-left: 24px;
        background: #009eff;
        box-shadow: 0px 2px 4px 0px rgba(0, 0, 0, 0.06);
        border-radius: 2px;
        border: 1px solid #009eff;
        font-family:
          PingFangSC,
          PingFang SC;
        font-weight: 400;
        font-size: 16px;
        color: #ffffff;
        cursor: pointer;
      }
    }
  }
  .exper-visualization {
    width: 100%;
    height: 950px;
    margin-top: 16px;
    background: #ffffff;
    box-shadow: 0px 2px 4px 0px rgba(0, 0, 0, 0.06);
    .visualization-title {
      display: flex;
      align-items: center;
      width: 100%;
      height: 45px;
      padding-left: 16px;
      border-bottom: 1px solid #eeeeee;
      color: #003d67;
      .visualization-name {
        height: 22px;
        margin-left: 8px;
        font-family:
          PingFangSC,
          PingFang SC;
        font-weight: 600;
        font-size: 16px;
        line-height: 22px;
      }
    }
    .visualization-main {
      display: flex;
      flex-direction: column;
      width: 100%;
      height: 904px;
      .visualization-tab {
        width: 362px;
        height: auto;
        margin: 20px auto 0 auto;
        .demo-tabs {
          :deep(.el-tabs__header) {
            margin: 0;
          }
          :deep(.el-tabs__item) {
            height: 24px;
            margin-bottom: 10px;
            padding: 0 48px 0 0;
            font-family:
              PingFangSC,
              PingFang SC;
            font-weight: 400;
            font-size: 16px;
            color: rgba(0, 0, 0, 0.65);
            line-height: 24px;
          }
          :deep(.el-tabs__item:nth-last-of-type(3)) {
            width: 170px;
          }
          :deep(.el-tabs__item:nth-last-of-type(1)) {
            padding: 0;
          }
          :deep(.el-tabs__item.is-active) {
            font-weight: 600;
            color: #2c406b;
          }
          :deep(.el-tabs__active-bar) {
            background-color: #003e6e;
          }
        }
      }
      .visualization-chart {
        // display: flex;
        width: 100%;
        margin-top: 20px;
        .chart-con {
          display: flex;
          justify-content: center;
          overflow: hidden;
          width: 100%;
          height: 800px;
          .chart-none{
            margin: 100px auto;
            font-size: 24px;
            color: #F56C6C;
          }
        }
        .chart-leng {
          display: flex;
          align-items: center;
          width: 168px;
          height: 22px;
          margin: -822px 32px 0 auto;
          .leng-left {
            display: flex;
            align-items: center;
            height: 22px;
            p {
              margin: 0;
              padding: 0;
            }
            p:nth-of-type(1) {
              width: 16px;
              height: 16px;
              margin-top: 1px;
              background: #009eff;
              border-radius: 50%;
            }
            p:nth-of-type(2) {
              height: 22px;
              margin-left: 8px;
              font-family:
                PingFangSC,
                PingFang SC;
              font-weight: 400;
              font-size: 16px;
              color: rgba(0, 0, 0, 0.65);
              line-height: 22px;
            }
          }
          .leng-right {
            display: flex;
            align-items: center;
            height: 22px;
            margin-left: 40px;
            p {
              margin: 0;
              padding: 0;
            }
            p:nth-of-type(1) {
              width: 16px;
              height: 16px;
              margin-top: 1px;
              background: #60d966;
              border-radius: 50%;
            }
            p:nth-of-type(2) {
              height: 22px;
              margin-left: 8px;
              font-family:
                PingFangSC,
                PingFang SC;
              font-weight: 400;
              font-size: 16px;
              color: rgba(0, 0, 0, 0.65);
              line-height: 22px;
            }
          }
        }
      }
      .visualization-error{
        margin: 100px auto;
        font-size: 24px;
        color: #F56C6C;
      }
    }
  }
}
:deep(.el-select__wrapper) {
  height: 36px;
  border-radius: 2px;
  font-family:
    PingFangSC,
    PingFang SC;
  font-size: 16px;
}
:deep(.el-select__input) {
  height: 28px;
}
:deep(.el-slider__button) {
  width: 12px;
  height: 12px;
  border: 1px solid #009eff;
  margin-top: -2px;
}
:deep(.el-slider__bar) {
  height: 4px;
  background-color: #009eff;
}
:deep(.el-slider__runway) {
  height: 4px;
}
:deep(.el-slider__marks-stop) {
  display: none;
}
:deep(.el-slider__marks-text) {
  margin-top: -26px;
  color: rgba(0, 0, 0, 0.85);
}
.upload-demo{
  width: 320px;
}
:deep(.el-upload-dragger){
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 0;
  height: 36px;
}
.el-upload__tip{
  display: flex;
  // justify-content: center;
  width: 320px;
  i{
    font-style: normal;
    color: #ff0000;
  }
}
</style>
