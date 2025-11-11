<template>
  <div class="product">
    <div v-if="detailValue === 0" class="product-main">
      <div class="title">测试详情</div>
      <div class="title-main">
        <div class="title-text">
          <p>测试名称</p>
          <p>{{ initObj.expName }}</p>
        </div>
        <div class="title-text">
          <p>算法名称</p>
          <p>{{ initObj.algorithmName }}</p>
        </div>
        <div class="title-text">
          <p>测试时间</p>
          <p>{{ initObj.createdAt }}</p>
        </div>
        <div class="title-back" @click="handleBack">返回</div>
      </div>
      <div class="evaluate-main">
        <div class="evaluate-title">
          <div class="i-svg:title"></div>
          <div class="evaluate-name">测试数据</div>
        </div>
        <div v-loading="dataLoading" class="evaluate-con" element-loading-text="Loading...">
          <div class="con-title">
            <p>数据名称</p>
            <p>{{ dataName }}</p>
          </div>
          <div class="con-line">
            <p></p>
            <p>投资者筛选</p>
          </div>
          <div class="evaluate-checkbox">
            <el-table :data="investorFilterData" class="checkbox-con">
              <el-table-column
                v-for="(item, index) in investorFilterColumns"
                :key="index"
                :prop="item.prop"
                :label="item.label"
                :width="item.width"
              ></el-table-column>
            </el-table>
          </div>
          <div class="con-line">
            <p></p>
            <p>产品筛选</p>
          </div>
          <div class="evaluate-checkbox">
            <el-table :data="financialFilterData" class="checkbox-con">
              <el-table-column
                v-for="(item, index) in financialFilterColumns"
                :key="index"
                :prop="item.prop"
                :label="item.label"
                :width="item.width"
              ></el-table-column>
            </el-table>
          </div>
        </div>
      </div>
      <div class="characteristic">
        <div class="evaluate-title">
          <div class="i-svg:title"></div>
          <div class="evaluate-name">特征选择</div>
        </div>
        <div class="investor-con">
          <p>投资者特征</p>
          <p>{{ investorCharacteristic }}</p>
        </div>
        <div class="financial-con">
          <p>金融产品特征</p>
          <p>{{ financialCharacteristic }}</p>
        </div>
      </div>
      <div class="characteristic" v-if="initObj.expTypeCode === 2">
        <div class="evaluate-title">
          <div class="i-svg:title"></div>
          <div class="evaluate-name">约束条件</div>
        </div>
        <div class="investor-con">
          <p>约束条件</p>
          <p>{{ constraintType }}</p>
        </div>
        <div class="financial-con">
          <p>约束条件值</p>
          <p>{{ constraintValue }}</p>
        </div>
      </div>
      <div class="characteristic" v-if="initObj.expTypeCode === 3">
        <div class="evaluate-title">
          <div class="i-svg:title"></div>
          <div class="evaluate-name">算法选择</div>
        </div>
        <div class="investor-con">
          <p>算法名称</p>
          <p>{{ initObj.algorithmName }}</p>
        </div>
      </div>
      <div class="evaluate-main" v-if="initObj.expTypeCode !== 4">
        <div class="evaluate-title">
          <div class="i-svg:title"></div>
          <div class="evaluate-name">{{ initObj.expTypeCode === 3 ? '算法对比结果':'算法性能评价' }}</div>
        </div>
        <div v-if="initObj.expTypeCode === 1" v-loading="algorithmLoading" class="evaluate-con" element-loading-text="Loading...">
          <div class="con-title">
            <p>算法名称</p>
            <p>{{ algorithm_name }}</p>
          </div>
          <div class="con-line">
            <p></p>
            <p>算法参数</p>
          </div>
          <div class="evaluate-checkbox">
            <el-table :data="evaluateData" class="checkbox-con">
              <el-table-column
                v-for="(item, index) in evaluateColumns"
                :key="index"
                :prop="item.prop"
                :label="item.label"
                :width="item.width"
              ></el-table-column>
            </el-table>
          </div>
          <div class="con-line">
            <p></p>
            <p>评价结果</p>
          </div>
          <div class="evaluate-checkbox">
            <el-table :data="resultData" class="checkbox-con">
              <el-table-column
                v-for="(item, index) in resultColumns"
                :key="index"
                :prop="item.prop"
                :label="item.label"
                :width="item.width"
              ></el-table-column>
            </el-table>
          </div>
        </div>
        <div v-if="initObj.expTypeCode === 2" v-loading="algorithmLoading" class="evaluate-constraint" element-loading-text="Loading...">
          <div class="evaluate-checkbox">
            <div v-for="(item, index) in constraintData" :key="index" class="chart-main">
              <div class="chart-title">
                <p></p>
                <p>{{ item.label }}</p>
              </div>
              <div class="chart-con">
                <Radar :id="'ggfxDb' + index" :chartObj="item.obj" v-if="item.obj.arr.length === 2" />
              </div>
            </div>
          </div>
        </div>
        <div v-if="initObj.expTypeCode === 3" v-loading="algorithmLoading" class="evaluate-newinvestors" element-loading-text="Loading...">
          <div class="evaluate-checkbox">
            <el-table :data="newinvestorsData" class="checkbox-con">
              <el-table-column
                v-for="(item, index) in newinvestorsColumns"
                :key="index"
                :prop="item.prop"
                :label="item.label"
                :width="item.width"
              ></el-table-column>
            </el-table>
          </div>
        </div>
      </div>
      <div class="result" v-if="initObj.expTypeCode === 1 || initObj.expTypeCode === 4">
        <div class="product-title">
          <div class="i-svg:title"></div>
          <div class="product-name">{{ initObj.expTypeCode === 1 ? '金融产品推荐结果' : '大模型解释'}}</div>
        </div>
        <div class="product-con">
          <div v-loading="loading" class="product-checkbox" element-loading-text="加载中...">
            <el-table :data="productData" class="checkbox-con">
              <el-table-column
                v-for="(item, index) in productColumns"
                :key="index"
                :prop="item.prop"
                :label="item.label"
                :width="item.width"
              >
                <template #default="scope">
                  <p
                    v-if="item.prop === 'action'"
                    class="check-action"
                    @click="handleClick(scope.row.user_no)"
                  >
                    {{ initObj.expTypeCode === 1 ? '详情' : '大模型解释'}}
                  </p>
                </template>
              </el-table-column>
            </el-table>
          </div>
        </div>
      </div>
    </div>
    <div v-if="detailValue === 1" class="detail">
      <div class="detail-title">投资者详情</div>
      <div v-if="detailAllData.base" class="detail-pho">
        <div class="pho-left">
          <div class="pho-img"></div>
          <div class="pho-con">
            <div class="con-name">{{ detailAllData.base.user_no || "" }}</div>
            <div class="con-list">
              <div class="list-text">
                性别：
                <i>{{ detailAllData.base.gender || "" }}</i>
              </div>
              <div class="list-text">
                教育背景：
                <i>{{ detailAllData.base.edu_level || "" }}</i>
              </div>
              <div class="list-text">
                毕业院校：
                <i>{{ detailAllData.base.university || "" }}</i>
              </div>
              <div class="list-text">
                年龄：
                <i>{{ detailAllData.base.age || "" }}</i>
              </div>
              <div class="list-text">
                职业：
                <i>{{ detailAllData.base.current_identity || "" }}</i>
              </div>
              <div class="list-text">
                投资次数：
                <i>{{ detailAllData.base.total_bid_number || "" }}</i>
              </div>
            </div>
          </div>
        </div>
        <div class="pho-right" @click="goList">返回</div>
      </div>
      <div class="detail-product" v-if="initObj.expTypeCode === 1">
        <div class="product-title">
          <div class="i-svg:title"></div>
          <div class="product-name">推荐金融产品列表</div>
        </div>
        <div class="product-list">
          <div
            v-if="
              detailAllData.recommended_products && detailAllData.recommended_products.length > 0
            "
            class="product-detail"
          >
            <div
              v-for="(item, index) in detailAllData.recommended_products"
              :key="index"
              class="financial"
            >
              <div class="financial-name">
                <p>{{ item.product_no }}</p>
                <p>{{ item.repayment }}</p>
              </div>
              <div class="financial-grade">
                <div class="grade-text">
                  <p>风险等级</p>
                  <p>{{ item.level }}</p>
                </div>
                <div class="grade-text">
                  <p>利率</p>
                  <p>{{ item.apr_percent }}</p>
                </div>
                <div class="grade-text">
                  <p>期限</p>
                  <p>{{ item.term }}个月</p>
                </div>
              </div>
              <div class="financial-scale">
                <p>规模：{{ item.total }}元</p>
                <p>被投资数：{{ item.bidders }}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="detail-investor" v-if="initObj.expTypeCode === 1">
        <div class="investor-title">
          <div class="i-svg:title"></div>
          <div class="investor-name">投资者偏好得分</div>
        </div>
        <div class="investor-chart">
          <div class="chart-left">
            <div class="chart-title">
              <p></p>
              <p>投资者特征得分</p>
            </div>
            <div class="chart-con">
              <Bar
                v-if="detailAllData.investor_features && detailAllData.investor_features.length > 0"
                id="investorChart"
                :chartData="detailAllData.investor_features"
              />
            </div>
          </div>
          <div class="chart-right">
            <div class="chart-title">
              <p></p>
              <p>产品特征得分</p>
            </div>
            <div class="chart-con">
              <Bar
                v-if="detailAllData.product_features && detailAllData.product_features.length > 0"
                id="scoreChart"
                :chartData="detailAllData.product_features"
              />
            </div>
          </div>
        </div>
      </div>
      <div class="detail-similar" v-if="initObj.expTypeCode === 1">
        <div class="similar-title">
          <div class="i-svg:title"></div>
          <div class="similar-name">最相似五位投资者</div>
        </div>
        <div
          v-if="detailAllData.similar_investors && detailAllData.similar_investors.length > 0"
          class="similar-list"
        >
          <div
            v-for="(item, index) in detailAllData.similar_investors"
            :key="index"
            class="similar-con"
            @click="handleClick(item.user_no)"
          >
            <div class="similar-img"></div>
            <div class="similar-text">
              <p>{{ item.user_no }}</p>
              <p>
                相似度
                <i>{{ (item.score * 100).toFixed(1) }}%</i>
              </p>
            </div>
          </div>
        </div>
      </div>
      <div class="detail-model" v-if="initObj.expTypeCode === 4">
        <div class="product-title">
          <div class="i-svg:title"></div>
          <div class="product-name">推荐金融产品</div>
        </div>
        <div class="product-list">
          <div
            v-if="
              detailAllData.llm_explains && detailAllData.llm_explains.length > 0
            "
            class="product-detail"
          >
            <div
              v-for="(item, index) in detailAllData.llm_explains"
              :key="index"
              class="financial"
            >
              <div class="financial-name">
                <div class="financial-title"><p>推荐产品</p></div>
                <p class="financial-text">{{ item.product_id }} ({{ item.tips }})</p>
              </div>
              <div class="financial-name financial-grade">
                <div class="financial-title"><p>大模型解释</p></div>
                <p class="financial-text">{{ item.explain }}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { debounce, cloneDeep } from "lodash-es";
import { ref, reactive, toRefs, onMounted, onBeforeUnmount, computed } from "vue";
import Radar from './radar/index.vue'
import Bar from "./bar/bar.vue";
import ZntgsyAPI from "@/api/zntgsy.api";
const emits = defineEmits(["goNextStep"]);
const props = defineProps({
  initData: {
    default: {},
    type: Object,
  },
});
const detailValue = ref(0);
const state = reactive({
  dataLoading: true,
  algorithmLoading: true,
  loading: true,
  initObj: cloneDeep(props.initData),
  algorithmTypeObj: {
    hidden_dim: "隐藏层维度",
    steps: "积分梯度计算步数",
    lr: "学习率",
    epochs: "训练轮数",
    iterations: "迭代次数",
    depth: "树的深度",
    opt_method: "优化方法",
    opt_options: "优化选项"
  },
  dataName: '',
  investorFilterObj: {
    gender: "性别",
    age: "年龄",
    edu_level: "教育程度",
    current_identity: "身份",
    user_invest_count: "投资次数"
  },
  investorFilterColumns: [],
  investorFilterData: [],
  financialFilterObj: {
    total: '规模',
    apr_percent: '利率',
    repayment: '还款方式',
    term: '借款期限',
    level: '风险等级',
    project_invest_count: '被投资次数'
  },
  financialFilterColumns: [],
  financialFilterData: [],
  investorCharacteristicObj: {
    CURRENTIDENTITY: '身份',
    University: '毕业院校',
    age: '年龄',
    baddebts_percent: '坏账比例',
    credit: '投资者信用',
    edu_level: '教育程度',
    gender: '性别',
    user_invest_count: '投资次数',
    weightedbidrate_percent: '加权投资利率'
  },
  investorCharacteristic: '',
  financialCharacteristicObj: {
    REPAYMENT: '还款方式',
    apr_percent: '利率',
    level: '风险等级',
    project_invest_count: '被投资次数',
    term: '借款期限',
    total: '规模'
  },
  financialCharacteristic: '',
  constraintType: '',
  constraintValue: '',
  constraintObj: {
    apr_percent: {
      name: '利率',
      arr: [
          {
              label: "前25%",
              value: 1
          },
          {
              label: "25%-50%",
              value: 2
          },
          {
              label: "50%-75%",
              value: 3
          },
          {
              label: "75%-100%",
              value: 4
          }
      ]
    },
    term: {
      name: '借款期限',
      arr: [
          {
              label: "前25%",
              value: 1
          },
          {
              label: "25%-50%",
              value: 2
          },
          {
              label: "50%-75%",
              value: 3
          },
          {
              label: "75%-100%",
              value: 4
          }
      ]
    },
    REPAYMENT: {
      name: '还款方式',
      arr: [
          {
              label: "等额本息",
              value: 0
          },
          {
              label: "一次性还本付息",
              value: 1
          },
          {
              label: "月还息，季还1/4本金",
              value: 2
          }
      ]
    },
    level: {
      name: '风险等级',
      arr: [
          {
              label: "1",
              value: 1
          },
          {
              label: "2",
              value: 2
          },
          {
              label: "3",
              value: 3
          },
          {
              label: "4",
              value: 4
          },
          {
              label: "5",
              value: 5
          },
          {
              label: "6",
              value: 6
          },
          {
              label: "7",
              value: 7
          },
          {
              label: "8",
              value: 8
          }
      ]
    },
  },
  evaluateColumns: [],
  evaluateData: [],
  resultColumns: [
    {
      prop: "accuracy",
      label: "准确率",
    },
    {
      prop: "precision",
      label: "查准率(精确率)",
    },
    {
      prop: "recall",
      label: "查全率(召回率)",
    },
    {
      prop: "f1",
      label: "F1值",
    },
    {
      prop: "auc",
      label: "AUC",
    },
  ],
  resultData: [],
  constraintData: [
    {
      label: '借款总额特征',
      value: 'total',
      obj: {
        arr: [],
        indicatorArr: []
      }
    },
    {
      label: '借款期限特征',
      value: 'term',
      obj: {
        arr: [],
        indicatorArr: []
      }
    },
    {
      label: '还款方式特征',
      value: 'REPAYMENT',
      obj: {
        arr: [],
        indicatorArr: []
      }
    },
    {
      label: '利率特征',
      value: 'apr_percent',
      obj: {
        arr: [],
        indicatorArr: []
      }
    },
    {
      label: '风险评分特征',
      value: 'level',
      obj: {
        arr: [],
        indicatorArr: []
      }
    },
    {
      label: '产品被投资次数特征',
      value: 'project_invest_count',
      obj: {
        arr: [],
        indicatorArr: []
      }
    }
  ],
  newinvestorsColumns: [
    {
      prop: "index",
      label: "序号",
      width: 120,
    },
    {
      prop: "Algorithm",
      label: "算法名称",
      width: 480,
    },
    {
      prop: "Weighted_HitRate",
      label: "Weighted_HitRate",
    },
  ],
  newinvestorsData: [],
  algorithm_name: props.initData.algorithmName,
  evaluateObj: {
    task_id: props.initData.expId,
    algorithm_type: props.initData.expTypeCode,
    algorithm_name: props.initData.algorithmType,
  },
  productColumns: [
    {
      label: "序号",
      prop: "index",
      width: 120,
    },
    {
      label: "投资者编号",
      prop: "user_no",
      width: 480,
    },
    {
      label: "推荐产品编号",
      prop: "recommend_products",
    },
    {
      label: "操作",
      prop: "action",
      width: 200,
    },
  ],
  productData: [],
  productObj: {
    task_id: props.initData.expId,
    algorithm_type: props.initData.expTypeCode,
    algorithm_name: props.initData.algorithmType,
    page_num: 1,
    page_size: 100000,
  },
  productTotal: 0,
  detailObj: {
    task_id: props.initData.expId,
    algorithm_type: props.initData.expTypeCode,
    algorithm_name: props.initData.algorithmType,
    investor_no: null,
  },
  detailAllData: {},
  explainsObj: {
    repayment: '还款方式',
    apr_percent: '利率',
    level: '风险等级',
    project_invest_count: '被投资次数',
    term: '借款期限',
    total: '规模'
  }
});
const { loading,
  dataLoading,
  algorithmLoading,
  initObj, 
  algorithmTypeObj,
  dataName,
  investorFilterObj,
  investorFilterColumns,
  investorFilterData,
  financialFilterColumns,
  financialFilterObj,
  financialFilterData,
  investorCharacteristicObj,
  investorCharacteristic,
  financialCharacteristicObj,
  financialCharacteristic,
  constraintType,
  constraintValue,
  constraintObj,
  evaluateColumns,
  evaluateData,
  resultColumns,
  resultData,
  constraintData,
  newinvestorsColumns,
  newinvestorsData,
  algorithm_name,
  evaluateObj,
  productColumns,
  productData,
  productObj,
  productTotal,
  detailObj,
  detailAllData,
  explainsObj } =
  toRefs(state);
onMounted(() => {
  getTaskParams();
  getTaskMetrics();
  if (state.initObj.expTypeCode === 1 || state.initObj.expTypeCode === 4) {
    getInvestors();
  }
});
const handleBack = () => {
  emits("goNextStep", {});
}
const getTaskParams = () => {
  // 获取入参参数
  state.dataName = ''
  state.investorFilterColumns = []
  state.investorFilterData = []
  state.financialFilterColumns = []
  state.financialFilterData = []
  state.dataLoading = true
  ZntgsyAPI.getTaskParams({ ...state.evaluateObj }).then((data) => {
    state.dataName = data.dataset_name
    if (data['filter']) {
      for (let key in data['filter']) {
        if (data['filter'].hasOwnProperty(key)) {
          if (typeof data['filter'][key] === 'object' && data['filter'][key] !== null) {
            data['filter'][key] = data['filter'][key].min + '~' + data['filter'][key].max;
          }
          if (state.investorFilterObj[key]) {
            state.investorFilterColumns.push({
              prop: key,
              label: state.investorFilterObj[key]
            })
          }
          if (state.financialFilterObj[key]) {
            state.financialFilterColumns.push({
              prop: key,
              label: state.financialFilterObj[key]
            })
          }
        }
      }
      state.investorFilterData.push(data['filter'])
      state.financialFilterData.push(data['filter'])
    }
    if (data.investor_features.length > 0) {
      let investorArr = data.investor_features.map((item) => {
        return state.investorCharacteristicObj[item]
      }).filter(item => item !== undefined)
      state.investorCharacteristic = investorArr.join('、')
    }
    if (data.product_features.length > 0) {
      let financialArr = data.product_features.map((item) => {
        return state.financialCharacteristicObj[item]
      }).filter(item => item !== undefined)
      state.financialCharacteristic = financialArr.join('、')
    }
    if (data.constraint_params) {
      for (const key in data.constraint_params) {
        if (Object.hasOwnProperty.call(data.constraint_params, key)) {
          state.constraintType = state.constraintObj[key].name
          let arr = []
          data.constraint_params[key].category.map((item) => {
            if (key === 'REPAYMENT') {
              arr.push(state.constraintObj[key].arr[item].label)
              state.constraintValue
            } else {
              arr.push(state.constraintObj[key].arr[item - 1].label)
            }
          })
          state.constraintValue = arr.join('、')
        }
      }
    }
    state.dataLoading = false
  }).catch((error) => {
    console.log(error)
  }) ;
}
const getTaskMetrics = () => {
  // 获取算法评价
  state.evaluateColumns = [];
  state.evaluateData = [];
  state.resultData = [];
  state.algorithmLoading = true
  ZntgsyAPI.getTaskMetrics({ ...state.evaluateObj }).then((data) => {
    if (data.algorithm_params) {
      for (let key in data.algorithm_params) {
        if (data.algorithm_params.hasOwnProperty(key)) {
          state.evaluateColumns.push({
            prop: key,
            label: state.algorithmTypeObj[key],
          });
        }
      }
      if (data.algorithm_params.opt_options) {
        let newStr = '';
        for (const key in data.algorithm_params.opt_options) {
          newStr += (key + '：' + data.algorithm_params.opt_options[key] + '；')
        }
        data.algorithm_params.opt_options = newStr;
      }
      state.evaluateData.push(data.algorithm_params);
    }
    if (state.initObj.expTypeCode === 1) {
      if (data.metrics) {
        state.resultData.push(data.metrics);
      }
    }
    if (state.initObj.expTypeCode === 2) {
      state.constraintData.map((item) => {
        if (item.value === 'REPAYMENT') {
          item.obj.arr = [
            {
              name: 'MAB',
              value: [data[item.value].mabtslr[1], data[item.value].mabtslr[2], data[item.value].mabtslr[3]],
              areaStyle: {
                  color: 'rgba(0, 158, 255, 0.15)' // 这个是关键
              },
              symbol: 'none', // 去掉点
            },
            {
              name: 'CMAB',
              value: [data[item.value].cmab[1], data[item.value].cmab[2], data[item.value].cmab[3]],
              areaStyle: {
                  color: 'rgba(255, 143, 0, 0.15)' // 这个是关键
              },
              symbol: 'none', // 去掉点
            },
          ]
          item.obj.indicatorArr = [
            {
              name: '等额本息',
              max: data[item.value].total[1]
            },
            {
              name: '一次性还本付息',
              max: data[item.value].total[2]
            },
            {
              name: '月还息，季还1/4本金',
              max: data[item.value].total[3]
            }
          ]
        } else if (item.value === 'level') {
          // const macArr = [69,null,25,40];
          item.obj.arr = [
            {
              name: 'MAB',
              value: [data[item.value].mabtslr[1], data[item.value].mabtslr[2], data[item.value].mabtslr[3], data[item.value].mabtslr[4], data[item.value].mabtslr[5], data[item.value].mabtslr[6], data[item.value].mabtslr[7], data[item.value].mabtslr[8]],
              areaStyle: {
                  color: 'rgba(0, 158, 255, 0.15)' // 这个是关键
              },
              symbol: 'none', // 去掉点
            },
            {
              name: 'CMAB',
              value: [data[item.value].cmab[1], data[item.value].cmab[2], data[item.value].cmab[3], data[item.value].cmab[4], data[item.value].cmab[5], data[item.value].cmab[6], data[item.value].cmab[7], data[item.value].cmab[8]],
              areaStyle: {
                  color: 'rgba(255, 143, 0, 0.15)' // 这个是关键
              },
              symbol: 'none', // 去掉点
            },
          ]
          item.obj.indicatorArr = [
            {
              name: 'Level 1',
              max: data[item.value].total[1]
            },
            {
              name: 'Level 2',
              max: data[item.value].total[2]
            },
            {
              name: 'Level 3',
              max: data[item.value].total[3] || ''
            },
            {
              name: 'Level 4',
              max: data[item.value].total[4]
            },
            {
              name: 'Level 5',
              max: data[item.value].total[5] || ''
            },
            {
              name: 'Level 6',
              max: data[item.value].total[6] || ''
            },
            {
              name: 'Level 7',
              max: data[item.value].total[7] || ''
            },
            {
              name: 'Level 8',
              max: data[item.value].total[8]
            }
          ]
        } else {
          item.obj.arr = [
            {
              name: 'MAB',
              value: [data[item.value].mabtslr[1], data[item.value].mabtslr[2], data[item.value].mabtslr[3], data[item.value].mabtslr[4]],
              areaStyle: {
                  color: 'rgba(0, 158, 255, 0.15)' // 这个是关键
              },
              symbol: 'none', // 去掉点
            },
            {
              name: 'CMAB',
              value: [data[item.value].cmab[1], data[item.value].cmab[2], data[item.value].cmab[3], data[item.value].cmab[4]],
              areaStyle: {
                  color: 'rgba(255, 143, 0, 0.15)' // 这个是关键
              },
              symbol: 'none', // 去掉点
            },
          ]
          item.obj.indicatorArr = [
            {
              name: 'Level 1',
              max: data[item.value].total[1]
            },
            {
              name: 'Level 2',
              max: data[item.value].total[2]
            },
            {
              name: 'Level 3',
              max: data[item.value].total[3]
            },
            {
              name: 'Level 4',
              max: data[item.value].total[4]
            }
          ]
        }
      })
    }
    if (state.initObj.expTypeCode === 3) {
      if (data.length > 0) {
        data.map((item, inde) => {
          item.index = inde + 1
        })
        state.newinvestorsData = data
      }
    }
    state.algorithmLoading = false
  });
};
const getInvestors = () => {
  // 获取已有算法
  state.loading = true;
  state.productData = [];
  ZntgsyAPI.getInvestors({ ...state.productObj }).then((data) => {
    if (state.initObj.expTypeCode === 1) {
      if (data.items.length > 0 ) {
        data.items.map((item, inde) => {
          state.productData.push({
            index: (state.productObj.page_num - 1) * state.productObj.page_size + inde + 1,
            user_no: item.user_no,
            recommend_products: item.recommend_products.join(";"),
          });
        });
        state.loading = false;
      } else {
        state.loading = false;
      }
    } else if (state.initObj.expTypeCode === 4) {
      if (data.length > 0) {
        data.map((item, inde) => {
          state.productData.push({
            index: (state.productObj.page_num - 1) * state.productObj.page_size + inde + 1,
            user_no: item.user_no,
            recommend_products: item.recommend_products.join(";"),
          });
        });
        state.loading = false;
      } else {
        state.loading = false;
      }
    }
  });
};
const handleCurrentChange = (val) => {
  state.productObj.page_num = val;
  getInvestors();
};
const handleClick = (id) => {
  getInvestorsDetail(id);
};
const getInvestorsDetail = (id) => {
  state.detailObj.investor_no = id - 0;
  // 获取投资者详情
  ZntgsyAPI.getInvestorsDetail({ ...state.detailObj }).then((data) => {
    state.detailAllData = data;
    state.detailAllData.llm_explains.map((item) => {
      let tipArr = [];
      Object.keys(item.item_info).forEach(key => {
        tipArr.push(state.explainsObj[key] + ':' + item.item_info[key])
      });
      item.tips = tipArr.join('，')
    })
    detailValue.value = 1;
  });
};
const goList = () => {
  detailValue.value = 0;
};
const handleStep = (num) => {
  emits("goNextStep", num);
};
const splitRadarValues = (values) => {
  const segments = [];
  let temp = [];
  values.forEach((v, i) => {
    if (v !== 0 && v != null) {
      temp.push({ i, v });
    } else if (temp.length) {
      segments.push(temp);
      temp = [];
    }
  });
  if (temp.length) segments.push(temp);

  // 转为 ECharts 需要的 value 数组
  return segments.map(seg => {
    const arr = new Array(values.length).fill(null);
    seg.forEach(({ i, v }) => (arr[i] = v));
    return arr;
  });
}
</script>

<style lang="scss" scoped>
.product {
  display: flex;
  flex-direction: column;
  width: 100%;
  background: #f6f6f6;
  .product-main {
    display: flex;
    flex-direction: column;
    width: calc(100% - 32px);
    min-height: 698px;
    padding-bottom: 16px;
    margin: 0 16px;
    .title{
      height: 40px;
      font-family: PingFangSC, PingFang SC;
      font-weight: 600;
      font-size: 18px;
      color: #003D67;
      line-height: 25px;
    }
    .title-main{
      display: flex;
      justify-content: space-between;
      width: 100%;
      height: 70px;
      .title-text{
        display: flex;
        flex-direction: column;
        p{
          margin: 0;
          padding: 0;
        }
        p:nth-of-type(1){
          height: 22px;
          font-family: PingFangSC, PingFang SC;
          font-weight: 400;
          font-size: 16px;
          color: rgba(0,0,0,0.45);
          line-height: 22px;
        }
        p:nth-of-type(2){
          height: 33px;
          margin-top: 8px;
          font-family: PingFangSC, PingFang SC;
          font-weight: 600;
          font-size: 24px;
          color: #02538A;
          line-height: 33px;
        }
      }
      .title-back{
        display: flex;
        justify-content: center;
        align-items: center;
        width: 86px;
        height: 36px;
        background: #FFFFFF;
        box-shadow: 0px 2px 4px 0px rgba(0,0,0,0.06);
        border-radius: 2px;
        border: 1px solid #CDCDCD;
        font-family: PingFangSC, PingFang SC;
        font-weight: 400;
        font-size: 16px;
        color: rgba(0,0,0,0.65);
        cursor: pointer;
      }
    }
    .evaluate-main {
      display: flex;
      flex-direction: column;
      width: 100%;
      min-height: 450px;
      margin-top: 16px;
      padding-bottom: 16px;
      background: #ffffff;
      box-shadow: 0px 2px 4px 0px rgba(0, 0, 0, 0.06);
      .evaluate-title {
        display: flex;
        align-items: center;
        width: 100%;
        height: 45px;
        padding-left: 16px;
        border-bottom: 1px solid #eeeeee;
        color: #003d67;
        .evaluate-name {
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
      .evaluate-con {
        display: flex;
        flex-direction: column;
        width: 100%;
        margin-top: 24px;
        padding: 0 24px;
        .con-title {
          display: flex;
          flex-direction: column;
          p {
            margin: 0;
            padding: 0;
          }
          p:nth-of-type(2) {
            height: 33px;
            font-family:
              PingFangSC,
              PingFang SC;
            font-weight: 600;
            font-size: 24px;
            color: #02538a;
            line-height: 33px;
          }
          p:nth-of-type(1) {
            height: 22px;
            font-family:
              PingFangSC,
              PingFang SC;
            font-weight: 400;
            font-size: 16px;
            color: rgba(0, 0, 0, 0.45);
            line-height: 22px;
          }
        }
        .con-line {
          display: flex;
          align-items: center;
          height: 22px;
          margin-top: 32px;
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
        }
        .evaluate-checkbox {
          display: flex;
          flex-direction: column;
          width: 100%;
          height: 114px;
          margin-top: 16px;
          .checkbox-con {
            width: 100%;
            height: 114;
            border-radius: 4px;
            overflow-x: hidden;
            :deep(.cell){
              text-align: center;
              padding: 0;
            }
          }
        }
        .evaluate-footer {
          display: flex;
          align-items: center;
          margin-top: 36px;
          p {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0;
            padding: 0;
            width: 156px;
            height: 36px;
            background: #ffffff;
            box-shadow: 0px 2px 4px 0px rgba(0, 0, 0, 0.06);
            border-radius: 2px;
            border: 1px solid #cdcdcd;
            font-family:
              PingFangSC,
              PingFang SC;
            font-weight: 400;
            font-size: 16px;
            color: rgba(0, 0, 0, 0.65);
            line-height: 56px;
            cursor: pointer;
          }
          p:nth-of-type(2) {
            margin-left: 24px;
            background: #009eff;
            border: 1px solid #009eff;
            color: #ffffff;
          }
        }
      }
      .evaluate-constraint{
        display: flex;
        flex-direction: column;
        width: 100%;
        // margin-top: 24px;
        padding: 0 24px;
        .evaluate-checkbox {
          display: flex;
          flex-wrap: wrap;
          width: 100%;
          height: auto;
          .chart-main{
            display: flex;
            flex-direction: column;
            width: 516px;
            height: 354px;
            margin-left: 26px;
            .chart-title{
              display: flex;
              align-items: center;
              width: 516px;
              height: 22px;
              margin-top: 20px;
              p{
                margin: 0;
                padding: 0;
              }
              p:nth-of-type(1){
                width: 2px;
                height: 14px;
                background: #003D67;
                border-radius: 1px;
              }
              p:nth-of-type(2){
                height: 22px;
                margin-left: 10px;
                font-family: PingFangSC, PingFang SC;
                font-weight: 600;
                font-size: 16px;
                color: #003D67;
                line-height: 22px;
              }
            }
            .chart-con {
              width: 516px;
              height: 300px;
              margin-top: 12px;
              background: #FBFBFB;
            }
          }
          .chart-main:nth-of-type(3n+1){
            margin-left: 0;
          }
        }
      }
      .evaluate-newinvestors{
        display: flex;
        flex-direction: column;
        width: 100%;
        // margin-top: 24px;
        padding: 0 24px;
        .evaluate-checkbox {
          display: flex;
          flex-direction: column;
          width: 100%;
          height: auto;
          margin-top: 24px;
          .checkbox-con {
            width: 100%;
            height: auto;
            border-radius: 4px;
            overflow-x: hidden;
          }
        }
      }
    }
    .characteristic{
      display: flex;
      flex-direction: column;
      width: 100%;
      max-height: 224px;
      margin-top: 16px;
      padding-bottom: 19px;
      background: #ffffff;
      box-shadow: 0px 2px 4px 0px rgba(0, 0, 0, 0.06);
      .evaluate-title {
        display: flex;
        align-items: center;
        width: 100%;
        height: 45px;
        padding-left: 16px;
        border-bottom: 1px solid #eeeeee;
        color: #003d67;
        .evaluate-name {
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
      .investor-con,
      .financial-con{
        display: flex;
        flex-direction: column;
        height: 55px;
        margin: 24px auto 0 24px;
        p{
          margin: 0;
          padding: 0;
        }
        p:nth-of-type(1){
          height: 22px;
          font-family: PingFangSC, PingFang SC;
          font-weight: 400;
          font-size: 16px;
          color: rgba(0,0,0,0.45);
          line-height: 22px;
        }
        p:nth-of-type(2){
          height: 25px;
          margin-top: 8px;
          font-family: PingFangSC, PingFang SC;
          font-weight: 600;
          font-size: 18px;
          color: #02538A;
          line-height: 25px;
        }
      }
    }
    .result{
      display: flex;
      flex-direction: column;
      width: 100%;
      height: 448px;
      margin-top: 16px;
      background: #ffffff;
      box-shadow: 0px 2px 4px 0px rgba(0, 0, 0, 0.06);
      .product-title {
        display: flex;
        align-items: center;
        width: 100%;
        height: 45px;
        padding-left: 16px;
        border-bottom: 1px solid #eeeeee;
        color: #003d67;
        .product-name {
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
      .product-con {
        display: flex;
        flex-direction: column;
        width: 100%;
        margin-top: 24px;
        padding: 0 24px;
        .con-name {
          display: flex;
          align-items: center;
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
        .product-checkbox {
          display: flex;
          flex-direction: column;
          width: 100%;
          height: auto;
          .checkbox-con {
            width: 100%;
            height: 354px;
            border-radius: 4px;
            overflow-x: hidden;
            .check-action {
              margin: 0;
              padding: 0;
              font-family:
                PingFangSC,
                PingFang SC;
              font-size: 16px;
              color: #009eff;
              cursor: pointer;
            }
          }
          .checkbox-page {
            display: flex;
            justify-content: end;
            width: 646px;
            margin: 10px 0 0 auto;
          }
        }
        .product-footer {
          display: flex;
          align-items: center;
          margin-top: 36px;
          p {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0;
            padding: 0;
            width: 156px;
            height: 36px;
            background: #ffffff;
            box-shadow: 0px 2px 4px 0px rgba(0, 0, 0, 0.06);
            border-radius: 2px;
            border: 1px solid #cdcdcd;
            font-family:
              PingFangSC,
              PingFang SC;
            font-weight: 400;
            font-size: 16px;
            color: rgba(0, 0, 0, 0.65);
            line-height: 56px;
            cursor: pointer;
          }
          p:nth-of-type(2) {
            margin-left: 24px;
            background: #009eff;
            border: 1px solid #009eff;
            color: #ffffff;
          }
        }
      }
    }
  }
  .detail {
    display: flex;
    flex-direction: column;
    width: calc(100% - 32px);
    height: auto;
    margin: 0 16px 20px 16px;
    .detail-title{
      width: 100%;
      height: 40px;
      font-family: PingFangSC, PingFang SC;
      font-weight: 600;
      font-size: 18px;
      color: #003D67;
    }
    .detail-pho {
      display: flex;
      justify-content: space-between;
      width: 100%;
      height: 108px;
      .pho-left {
        display: flex;
        height: 108px;
        .pho-img {
          width: 100px;
          height: 108px;
          background: url("@/assets/images/product_icon.png") center center no-repeat;
          border-radius: 2px;
        }
        .pho-con {
          display: flex;
          flex-direction: column;
          margin-left: 24px;
          .con-name {
            height: 28px;
            font-family:
              PingFangSC,
              PingFang SC;
            font-weight: 600;
            font-size: 20px;
            color: #003d67;
            line-height: 28px;
          }
          .con-list {
            display: flex;
            flex-wrap: wrap;
            width: 700px;
            margin-top: 20px;
            .list-text {
              height: 22px;
              margin-bottom: 16px;
              font-family:
                PingFangSC,
                PingFang SC;
              font-weight: 400;
              font-size: 16px;
              color: rgba(0, 0, 0, 0.65);
              line-height: 22px;
              i {
                font-style: normal;
                height: 22px;
                font-family:
                  PingFangSC,
                  PingFang SC;
                font-weight: 400;
                font-size: 16px;
                color: rgba(0, 0, 0, 0.85);
                line-height: 22px;
              }
            }
            .list-text:nth-of-type(1) {
              width: 188px;
            }
            .list-text:nth-of-type(2) {
              width: 248px;
            }
            .list-text:nth-of-type(3) {
              width: 232px;
            }
            .list-text:nth-of-type(4) {
              width: 220px;
            }
            .list-text:nth-of-type(5) {
              width: 216px;
            }
            .list-text:nth-of-type(6) {
              width: 188px;
            }
          }
        }
      }
      .pho-right {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 86px;
        height: 36px;
        background: #ffffff;
        box-shadow: 0px 2px 4px 0px rgba(0, 0, 0, 0.06);
        border-radius: 2px;
        border: 1px solid #cdcdcd;
        font-family:
          PingFangSC,
          PingFang SC;
        font-weight: 400;
        font-size: 16px;
        color: rgba(0, 0, 0, 0.65);
        line-height: 36px;
        cursor: pointer;
      }
    }
    .detail-product {
      display: flex;
      flex-direction: column;
      width: 100%;
      height: 256px;
      margin-top: 24px;
      background: #ffffff;
      box-shadow: 0px 2px 4px 0px rgba(0, 0, 0, 0.06);
      .product-title {
        display: flex;
        align-items: center;
        width: 100%;
        height: 45px;
        padding-left: 16px;
        border-bottom: 1px solid #eeeeee;
        color: #003d67;
        .product-name {
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
      .product-list {
        overflow-y: hidden;
        overflow-x: auto;
        width: 100%;
        height: 186px;
        margin: 20px 0 0 0;
        padding: 0 24px;
        .product-detail {
          display: flex;
          width: auto;
          height: 166px;
          .financial {
            display: flex;
            flex-direction: column;
            width: 308px;
            height: 166px;
            border-radius: 2px;
            background: #f4faff;
            border: 1px solid #c5e5ff;
            .financial-name {
              display: flex;
              width: 308px;
              height: 44px;
              padding: 6px 8px 6px 20px;
              background: linear-gradient(180deg, #d6edff 0%, rgba(225, 241, 255, 0) 100%);
              p {
                margin: 0;
                padding: 0;
              }
              p:nth-of-type(1) {
                height: 24px;
                margin-top: 4px;
                font-family:
                  PingFangSC,
                  PingFang SC;
                font-weight: 600;
                font-size: 18px;
                color: #003d67;
                line-height: 24px;
              }
              p:nth-of-type(2) {
                display: flex;
                justify-content: center;
                align-items: center;
                // width: 76px;
                height: 32px;
                padding: 0 10px;
                margin-left: auto;
                background: #fffbf2;
                border-radius: 2px;
                border: 1px solid #eae1c7;
                font-family:
                  PingFangSC,
                  PingFang SC;
                font-weight: 600;
                font-size: 14px;
                color: #a88433;
              }
            }
            .financial-grade {
              display: flex;
              width: 268px;
              height: 60px;
              margin: 12px 20px 0 20px;
              border-bottom: 1px solid #d0dee9;
              .grade-text {
                display: flex;
                flex-direction: column;
                p {
                  margin: 0;
                  padding: 0;
                }
                p:nth-of-type(1) {
                  height: 20px;
                  font-family:
                    PingFangSC,
                    PingFang SC;
                  font-weight: 400;
                  font-size: 14px;
                  color: rgba(0, 0, 0, 0.65);
                  line-height: 20px;
                }
                p:nth-of-type(2) {
                  height: 22px;
                  margin-top: 8px;
                  font-family:
                    PingFangSC,
                    PingFang SC;
                  font-weight: 600;
                  font-size: 16px;
                  line-height: 22px;
                }
              }
              .grade-text:nth-of-type(1) {
                p:nth-of-type(2) {
                  color: #eea91a;
                }
              }
              .grade-text:nth-of-type(2) {
                margin-left: 52px;
                p:nth-of-type(2) {
                  color: #ff0000;
                }
              }
              .grade-text:nth-of-type(3) {
                margin-left: 58px;
                p:nth-of-type(2) {
                  color: #ff0000;
                }
              }
            }
            .financial-scale {
              display: flex;
              align-items: center;
              width: 268px;
              height: 20px;
              margin: 12px 20px 0 20px;
              p {
                margin: 0;
                padding: 0;
                height: 20px;
                font-family:
                  PingFangSC,
                  PingFang SC;
                font-weight: 400;
                font-size: 14px;
                color: rgba(0, 0, 0, 0.65);
                line-height: 20px;
                i {
                  font-style: normal;
                  color: rgba(0, 0, 0, 0.85);
                }
              }
              p:nth-of-type(1) {
                width: 170px;
              }
              p:nth-of-type(2) {
                width: 98px;
              }
            }
          }
          .financial:nth-of-type(n + 2) {
            margin-left: 16px;
          }
        }
      }
    }
    .detail-investor {
      display: flex;
      flex-direction: column;
      width: 100%;
      height: 500px;
      margin-top: 16px;
      background: #ffffff;
      box-shadow: 0px 2px 4px 0px rgba(0, 0, 0, 0.06);
      .investor-title {
        display: flex;
        align-items: center;
        width: 100%;
        height: 45px;
        padding-left: 16px;
        border-bottom: 1px solid #eeeeee;
        color: #003d67;
        .investor-name {
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
      .investor-chart {
        display: flex;
        justify-content: space-between;
        margin: 16px 24px 24px 24px;
        .chart-left,
        .chart-right {
          display: flex;
          flex-direction: column;
          width: 786px;
          height: 414px;
          .chart-title {
            display: flex;
            align-items: center;
            width: 786px;
            height: 22px;
            p {
              margin: 0;
              padding: 0;
            }
            p:nth-of-type(1) {
              width: 2px;
              height: 14px;
              margin-top: 1px;
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
          }
          .chart-con {
            width: 786px;
            height: 376px;
            margin-top: 16px;
          }
        }
      }
    }
    .detail-similar {
      display: flex;
      flex-direction: column;
      width: 100%;
      height: 180px;
      margin-top: 16px;
      background: #ffffff;
      box-shadow: 0px 2px 4px 0px rgba(0, 0, 0, 0.06);
      .similar-title {
        display: flex;
        align-items: center;
        width: 100%;
        height: 45px;
        padding-left: 16px;
        border-bottom: 1px solid #eeeeee;
        color: #003d67;
        .similar-name {
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
      .similar-list {
        display: flex;
        justify-content: space-between;
        width: 100%;
        height: 94px;
        margin: 16px 0 24px 0;
        padding: 0 24px;
        .similar-con {
          display: flex;
          width: 304px;
          height: 94px;
          background: #f4faff;
          border-radius: 2px;
          cursor: pointer;
          .similar-img {
            width: 80px;
            height: 86px;
            margin: 4px;
            background: url("@/assets/images/product_icon1.png") center center no-repeat;
            border-radius: 2px;
          }
          .similar-text {
            display: flex;
            flex-direction: column;
            margin-top: 4px;
            p {
              margin: 0;
              padding: 0;
            }
            p:nth-of-type(1) {
              width: 212px;
              height: 44px;
              padding-left: 16px;
              background: #e7f1ff;
              border-radius: 2px;
              font-family:
                PingFangSC,
                PingFang SC;
              font-weight: 600;
              font-size: 18px;
              color: #003d67;
              line-height: 44px;
            }
            p:nth-of-type(2) {
              width: 212px;
              height: 22px;
              margin-top: 13px;
              padding-left: 16px;
              font-family:
                PingFangSC,
                PingFang SC;
              font-weight: 400;
              font-size: 14px;
              color: rgba(0, 0, 0, 0.65);
              line-height: 22px;
              i {
                height: 22px;
                margin-left: 12px;
                font-style: normal;
                font-family:
                  PingFangSC,
                  PingFang SC;
                font-weight: 600;
                font-size: 16px;
                color: #ff0000;
                line-height: 22px;
              }
            }
          }
        }
      }
    }
    .detail-model {
      display: flex;
      flex-direction: column;
      width: 100%;
      height: auto;
      margin-top: 24px;
      padding-bottom: 20px;
      background: #ffffff;
      box-shadow: 0px 2px 4px 0px rgba(0, 0, 0, 0.06);
      .product-title {
        display: flex;
        align-items: center;
        width: 100%;
        height: 45px;
        padding-left: 16px;
        border-bottom: 1px solid #eeeeee;
        color: #003d67;
        .product-name {
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
      .product-list {
        display: flex;
        flex-direction: column;
        width: 100%;
        height: auto;
        margin: 0;
        padding: 0 32px;
        .product-detail {
          display: flex;
          flex-direction: column;
          width: 100%;
          height: auto;
          .financial {
            display: flex;
            flex-direction: column;
            width: 100%;
            height: auto;
            margin: 20px 0 4px 0;
            .financial-name {
              display: flex;
              width: 100%;
              height: auto;
              p {
                margin: 0;
                padding: 0;
              }
              .financial-title {
                display: flex;
                justify-content: center;
                align-items: center;
                width: 48px;
                height: 48px;
                background: linear-gradient( 180deg, #00CDFF 0%, #009EFF 100%);
                font-family: PingFangSC, PingFang SC;
                font-weight: 600;
                font-size: 12px;
                border-radius: 50%;
                color: #FFFFFF;
                line-height: 14px;
                p{
                  width: 24px;
                }
              }
              .financial-text {
                display: flex;
                justify-content: center;
                align-items: center;
                max-width: 1524px;
                height: 36px;
                margin: 6px auto 0 12px;
                padding: 8px 12px;
                background: #F3F9FF;
                border-radius: 4px;
                font-family: PingFangSC, PingFang SC;
                font-weight: 400;
                font-size: 14px;
                color: rgba(0,0,0,0.85);
                line-height: 20px;
              }
            }
            .financial-grade {
              margin-top: 16px;
              .financial-title{
                p{
                  width: 36px;
                  text-align: center;
                }
              }
              .financial-text{
                height: auto;
                margin: 0 auto 0 12px;
              }
            }
          }
        }
      }
    }
  }
}
:deep(.el-table .cell) {
  font-family:
    PingFangSC,
    PingFang SC;
  padding: 0 0 0 12px;
  font-size: 16px;
  overflow: hidden; /* 超出隐藏 */
  white-space: nowrap; /* 不换行 */
  text-overflow: ellipsis; /* 显示省略号 */
}
:deep(.el-table th.el-table__cell .cell) {
  height: 54px;
  line-height: 54px;
}
:deep(.el-table td.el-table__cell .cell) {
  height: 60px;
  line-height: 60px;
  border-bottom: 1px solid #e8e8e8;
}
:deep(.el-table th.el-table__cell) {
  background: #f4faff;
  color: rgba(0, 0, 0, 0.85);
  border-bottom: none;
  height: 54px;
  padding: 0;
}
:deep(.el-table--enable-row-hover .el-table__body .el-table__row:hover > td.el-table__cell) {
  background: transparent;
}
:deep(.el-table th.el-table__cell:nth-last-of-type(n + 2) .cell) {
  border-right: 1px solid #e8e8e8;
}
:deep(.el-table td.el-table__cell) {
  height: 59px;
  padding: 0;
  border-bottom: none;
}
:deep(.el-table td.el-table__cell:nth-last-of-type(n + 2) .cell) {
  border-right: 1px solid #e8e8e8;
}
:deep(.el-table__inner-wrapper:before) {
  background-color: transparent;
}
</style>
