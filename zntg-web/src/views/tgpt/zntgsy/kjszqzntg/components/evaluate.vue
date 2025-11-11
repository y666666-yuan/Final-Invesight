<template>
  <div class="product">
    <div v-if="detailValue === 0" class="product-main">
      <div class="product-title">
        <div class="i-svg:title"></div>
        <div class="product-name">推荐结果及解释</div>
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
                  大模型解释
                </p>
              </template>
            </el-table-column>
          </el-table>
          <!-- <div v-if="productTotal > 0" class="checkbox-page">
            <el-pagination
              v-model:current-page="productObj.page_num"
              v-model:page-size="productObj.page_size"
              layout="total,prev, pager, next, jumper"
              :total="productTotal"
              @current-change="handleCurrentChange"
            />
          </div> -->
        </div>
        <div class="product-footer">
          <p @click="handleStep(4)">上一步</p>
          <p @click="handleResult">完成</p>
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
      <div class="detail-product">
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
    <!-- 完成之后的提示 -->
    <el-dialog
      v-model="dialogVisible"
      width="504"
      title=""
      align-center
      :show-close="false"
    >
    <div class="result-main">
      <div class="result-title">
          <div class="i-svg:title"></div>
          <div>保存测试结果</div>
          <el-icon class="result-icon" @click="closeResult(1)"><Close /></el-icon>
        </div>
      <div class="result-con">
        <div class="result-text">
          请确认测试结果名称，如不需要保存请点击【不保存】
        </div>
        <div class="result-name">
          <p>测试结果名称：</p>
          <el-input v-model="resultObj.expName" />
        </div>
      </div>
      <div class="result-footer">
        <p @click="closeResult(2)">不保存</p>
        <p @click="successResult">保存</p>
      </div>
    </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { debounce, cloneDeep } from "lodash-es";
import { ref, reactive, toRefs, onMounted, onBeforeUnmount, computed } from "vue";
import Bar from "./bar/index.vue";
import ZntgsyAPI from "@/api/zntgsy.api";
import SyjgckAPI from "@/api/syjgck.api";
const emits = defineEmits(["goNextStep"]);
const props = defineProps({
  taskId: {
    default: "",
    type: String,
  },
  algorithmType: {
    default: "",
    type: String,
  },
  algorithmName: {
    default: "",
    type: String,
  },
  data: {
    default: [],
    type: Array,
  },
});
const detailValue = ref(0);
const state = reactive({
  loading: true,
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
    task_id: props.taskId,
    algorithm_type: 4,
    algorithm_name: props.algorithmType,
    page_num: 1,
    page_size: 100000,
  },
  productTotal: 0,
  dialogVisible: false,
  detailObj: {
    task_id: props.taskId,
    algorithm_type: 4,
    algorithm_name: props.algorithmType,
    investor_no: null,
  },
  detailAllData: {},
  resultObj: {
    expId: props.taskId,
    algorithmName: props.algorithmName,
    algorithmType: props.algorithmType,
    expTypeCode: 4,
    expTypeName: '可解释增强的智能投顾',
    expName: ''
  },
  financialCharacteristicObj: {
    repayment: '还款方式',
    apr_percent: '利率',
    level: '风险等级',
    project_invest_count: '被投资次数',
    term: '借款期限',
    total: '规模'
  }
});
const { loading, productColumns, productData, productObj, productTotal, dialogVisible, detailObj, detailAllData, resultObj, financialCharacteristicObj } =
  toRefs(state);
onMounted(() => {
  getInvestors();
});
const getInvestors = () => {
  // 获取已有算法
  state.loading = true;
  state.productData = [];
  ZntgsyAPI.getInvestors({ ...state.productObj }).then((data) => {
    if (data.length > 0) {
      data.map((item, inde) => {
        state.productData.push({
          index: (state.productObj.page_num - 1) * state.productObj.page_size + inde + 1,
          user_no: item.user_no,
          recommend_products: item.recommend_products.join(";"),
        });
      });
      // state.productTotal = data.total;
      state.loading = false;
    } else {
      state.loading = false;
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
        tipArr.push(state.financialCharacteristicObj[key] + ':' + item.item_info[key])
      });
      item.tips = tipArr.join('，')
    })
    detailValue.value = 1;
  });
};
const goList = () => {
  detailValue.value = 0;
};
const handleResult = () => {
  state.dialogVisible = true
}
const closeResult = (num) => {
  if (num === 1) {
    state.dialogVisible = false
  } else {
    state.dialogVisible = false
    emits("goNextStep", 1, "down")
  }
}
const successResult = () => {
  if (state.resultObj.expName) {
    SyjgckAPI.postSaveExp({ ...state.resultObj }).then((data) => {
      if (data && data.expId) {
        ElMessage.success('测试结果保存成功')
        state.dialogVisible = false
        emits("goNextStep", 1, "down")
      }
    });
  } else {
    ElMessage.warning('请填写实验结果名称');
  }
}
const handleStep = (num) => {
  emits("goNextStep", num);
};
</script>

<style lang="scss" scoped>
.product {
  display: flex;
  flex-direction: column;
  width: 100%;
  background: #f6f6f6;
  .product-main {
    width: 100%;
    min-height: 698px;
    padding-bottom: 16px;
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
  .detail {
    display: flex;
    flex-direction: column;
    width: 100%;
    height: auto;
    margin-top: 8px;
    .detail-title{
      height: 25px;
      font-family: PingFangSC, PingFang SC;
      font-weight: 600;
      font-size: 18px;
      color: #003D67;
      line-height: 25px;
    }
    .detail-pho {
      display: flex;
      justify-content: space-between;
      width: 100%;
      height: 108px;
      margin-top: 15px;
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
:deep(.el-dialog){
  padding: 0;
}
:deep(.el-dialog__header){
  display: none;
}
.result-main{
  display: flex;
  flex-direction: column;
  width: 504px;
  height: 260px;
  .result-title{
    display: flex;
    align-items: center;
    width: 504px;
    height: 45px;
    padding: 0 20px 0 16px;
    border-bottom: 1px solid #EEEEEE;
    font-family: PingFangSC, PingFang SC;
    font-weight: 600;
    font-size: 16px;
    color: #003D67;
    div:nth-of-type(1){
      margin-top: 2px;
      font-size: 16px;
    }
    div:nth-of-type(2){
      margin-left: 8px;
    }
    .result-icon{
      margin-left: auto;
      font-size: 16px;
      color: #ABABAB;
      cursor: pointer;
    }
  }
  .result-con{
    display: flex;
    flex-direction: column;
    width: 384px;
    height: auto;
    margin: 32px auto 0 auto;
    .result-text{
      width: 384px;
      height: 22px;
      font-family: PingFangSC, PingFang SC;
      font-weight: 400;
      font-size: 16px;
      color: rgba(0,0,0,0.85);
    }
    .result-name{
      display: flex;
      align-items: center;
      width: 384px;
      height: 32px;
      margin-top: 24px;
      p{
        width: 112px;
        height: 32px;
        margin: 0;
        padding: 0;
        line-height: 30px;
        font-family: PingFangSC, PingFang SC;
        font-weight: 400;
        font-size: 16px;
        color: rgba(0,0,0,0.85);
      }
      :deep(.el-input){
          width: 272px !important;
        }
    }
  }
  .result-footer{
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 32px;
    p{
      margin: 0;
      padding: 0;
    }
    p:nth-of-type(1){
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
    p:nth-of-type(2){
      display: flex;
      justify-content: center;
      align-items: center;
      width: 86px;
      height: 36px;
      margin-left: 24px;
      background: #009EFF;
      box-shadow: 0px 2px 4px 0px rgba(0,0,0,0.06);
      border-radius: 2px;
      border: 1px solid #009EFF;
      font-family: PingFangSC, PingFang SC;
      font-weight: 400;
      font-size: 16px;
      color: #FFFFFF;
      cursor: pointer;
    }
  }
}
</style>
