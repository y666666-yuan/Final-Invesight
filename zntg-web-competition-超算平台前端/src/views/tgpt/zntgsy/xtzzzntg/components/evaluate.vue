<template>
  <div class="evaluate">
    <div class="evaluate-main">
      <div class="evaluate-title">
        <div class="i-svg:title"></div>
        <div class="evaluate-name">算法对比结果</div>
      </div>
      <div v-loading="loading" class="evaluate-con" element-loading-text="Loading...">
        <!-- <div class="con-title">
          <p>算法名称</p>
          <p>{{ algorithm_name }}</p>
        </div>
        <div class="con-line">
          <p></p>
          <p>算法参数</p>
        </div> -->
        <!-- <div class="evaluate-checkbox">
          <el-table :data="evaluateData" class="checkbox-con">
            <el-table-column
              v-for="(item, index) in evaluateColumns"
              :key="index"
              :prop="item.prop"
              :label="item.label"
              :width="item.width"
            ></el-table-column>
          </el-table>
        </div> -->
        <!-- <div class="con-line">
          <p></p>
          <p>评价结果</p>
        </div> -->
        <div class="evaluate-checkbox">
          <el-table :data="resultData" class="checkbox-con">
            <el-table-column
              v-for="(item, index) in resultColumns"
              :key="index"
              :prop="item.prop"
              :label="item.label"
              :width="item.width"
            >
              <template #header>
                <div v-if="item.prop === 'Weighted_HitRate'" class="evaluate-wei">
                  <span>加权命中率</span>
                  <el-tooltip
                    effect="light"
                    placement="top"
                  >
                    <template #content>推荐产品若与用户真实交易记录匹配，则按排名赋予权重（在推荐的5个<br />产品当中所处排名越高，权重越大），最终加权求和以量化推荐质量。</template>
                    <el-icon class="chart-icon"><Warning /></el-icon>
                  </el-tooltip>
                </div>
              </template>
            </el-table-column>
          </el-table>
        </div>
        <div class="evaluate-footer">
          <p @click="handleStep(4)">上一步</p>
          <p @click="handleResult">完成</p>
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
const state = reactive({
  loading: false,
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
  evaluateColumns: [],
  evaluateData: [],
  resultColumns: [
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
      label: "加权命中率",
    },
    // {
    //   prop: "f1",
    //   label: "F1值",
    // },
    // {
    //   prop: "auc",
    //   label: "AUC",
    // },
  ],
  resultData: [],
  algorithm_name: props.algorithmName,
  evaluateObj: {
    task_id: props.taskId,
    algorithm_type: 3,
    algorithm_name: props.algorithmType,
  },
  dialogVisible: false,
  resultObj: {
    expId: props.taskId,
    algorithmName: props.algorithmName,
    algorithmType: props.algorithmType,
    expTypeCode: 3,
    expTypeName: '对新投资者的智能投顾',
    expName: ''
  }
});
const {
  loading,
  algorithmTypeObj,
  evaluateColumns,
  evaluateData,
  resultColumns,
  resultData,
  algorithm_name,
  evaluateObj,
  dialogVisible,
  resultObj
} = toRefs(state);
onMounted(() => {
  getTaskMetrics();
});
const getTaskMetrics = () => {
  // 获取已有算法
  state.loading = true;
  state.evaluateColumns = [];
  state.evaluateData = [];
  state.resultData = [];
  ZntgsyAPI.getTaskMetrics({ ...state.evaluateObj }).then((res) => {
    // if (data.algorithm_params) {
    //   for (let key in data.algorithm_params) {
    //     if (data.algorithm_params.hasOwnProperty(key)) {
    //       state.evaluateColumns.push({
    //         prop: key,
    //         label: state.algorithmTypeObj[key],
    //       });
    //     }
    //   }
    //   if (data.algorithm_params.opt_options) {
    //     let newStr = '';
    //     for (const key in data.algorithm_params.opt_options) {
    //       newStr += (key + '：' + data.algorithm_params.opt_options[key] + '；')
    //     }
    //     data.algorithm_params.opt_options = newStr;
    //   }
    //   state.evaluateData.push(data.algorithm_params);
    // }
    if (res.length > 0) {
      res.map((item, inde) => {
        item.index = inde + 1
      })
      state.resultData = res
    }
    state.loading = false;
  });
};
const handleStep = (num) => {
  if (num === 4) {
    emits("goNextStep", num, "up");
  } else {
    // ElMessage.success('实验已完成');
    emits("goNextStep", num, "down");
  }
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
</script>

<style lang="scss" scoped>
.evaluate {
  display: flex;
  flex-direction: column;
  width: 100%;
  background: #f6f6f6;
  .evaluate-main {
    width: 100%;
    height: 698px;
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
        height: auto;
        margin-top: 16px;
        .checkbox-con {
          width: 100%;
          height: auto;
          border-radius: 4px;
          overflow-x: hidden;
          .evaluate-wei{
            display: flex !important;
            justify-content: center;
            align-items: center;
            width: 1000px;
          }
          .chart-icon{
            margin: 2px 0 0 4px;
            line-height: 20px;
            transform: rotate(180deg);
            color: #969696;
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
  }
}
:deep(.el-table .cell) {
  font-family:
    PingFangSC,
    PingFang SC;
  padding: 0;
  font-size: 16px;
  overflow: hidden; /* 超出隐藏 */
  white-space: nowrap; /* 不换行 */
  text-overflow: ellipsis; /* 显示省略号 */
  text-align: center;
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
