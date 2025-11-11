<template>
  <div class="evaluate">
    <div class="evaluate-main">
      <div class="evaluate-title">
        <div class="i-svg:title"></div>
        <div class="evaluate-name">算法性能评价</div>
      </div>
      <div v-loading="loading" class="evaluate-con" element-loading-text="Loading...">
        <div class="evaluate-checkbox">
          <div v-for="(item, index) in resultData" :key="index" class="chart-main">
            <div class="chart-title">
              <p></p>
              <p>{{ item.label }}</p>
              <el-tooltip
                class="box-item"
                effect="light"
                :content="item.content"
                placement="top"
              >
              <el-icon class="chart-icon"><Warning /></el-icon>
              </el-tooltip>
            </div>
            <div class="chart-con">
              <Radar :id="'ggfxDb' + index" :chartObj="item.obj" v-if="item.obj.arr.length === 2" />
            </div>
          </div>
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
import Radar from './radar/index.vue'
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
      label: "Weighted_HitRate",
    },
  ],
  resultData: [
    {
      label: '借款总额特征',
      value: 'total',
      content: '按照相关特征升序排序，前25%；25%-50%；50%-75%；75%-100%',
      obj: {
        arr: [],
        indicatorArr: []
      }
    },
    {
      label: '借款期限特征',
      value: 'term',
      content: '按照相关特征升序排序，前25%；25%-50%；50%-75%；75%-100%',
      obj: {
        arr: [],
        indicatorArr: []
      }
    },
    {
      label: '还款方式特征',
      value: 'REPAYMENT',
      content: '0：等额本息，1：一次性还本付息，2：月还息，季还1/4本金',
      obj: {
        arr: [],
        indicatorArr: []
      }
    },
    {
      label: '利率特征',
      value: 'apr_percent',
      content: '按照相关特征升序排序，前25%；25%-50%；50%-75%；75%-100%',
      obj: {
        arr: [],
        indicatorArr: []
      }
    },
    {
      label: '风险评分特征',
      value: 'level',
      content: '1-8，风险依次上升',
      obj: {
        arr: [],
        indicatorArr: []
      }
    },
    {
      label: '产品被投资次数特征',
      value: 'project_invest_count',
      content: '按照相关特征升序排序，前25%；25%-50%；50%-75%；75%-100%',
      obj: {
        arr: [],
        indicatorArr: []
      }
    }
  ],
  algorithm_name: props.algorithmName,
  
  evaluateObj: {
    task_id: props.taskId,
    algorithm_type: 2,
    algorithm_name: props.algorithmType,
  },
  dialogVisible: false,
  resultObj: {
    expId: props.taskId,
    algorithmName: props.algorithmName,
    algorithmType: props.algorithmType,
    expTypeCode: 2,
    expTypeName: '带约束条件的智能投顾实验',
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
  ZntgsyAPI.getTaskMetrics({ ...state.evaluateObj }).then((res) => {
    if (res) {
      state.resultData.map((item) => {
        if (item.value === 'REPAYMENT') {
          item.obj.arr = [
            {
              name: 'MAB',
              value: [res[item.value].mabtslr[1], res[item.value].mabtslr[2], res[item.value].mabtslr[3]],
              areaStyle: {
                  color: 'rgba(0, 158, 255, 0.15)' // 这个是关键
              },
              symbol: 'none', // 去掉点
            },
            {
              name: 'CMAB',
              value: [res[item.value].cmab[1], res[item.value].cmab[2], res[item.value].cmab[3]],
              areaStyle: {
                  color: 'rgba(255, 143, 0, 0.15)' // 这个是关键
              },
              symbol: 'none', // 去掉点
            },
          ]
          item.obj.indicatorArr = [
            {
              name: '等额本息',
              max: res[item.value].total[1]
            },
            {
              name: '一次性还本付息',
              max: res[item.value].total[2]
            },
            {
              name: '月还息，季还1/4本金',
              max: res[item.value].total[3]
            }
          ]
        } else if (item.value === 'level') {
          item.obj.arr = [
            {
              name: 'MAB',
              value: [res[item.value].mabtslr[1], res[item.value].mabtslr[2], res[item.value].mabtslr[3], res[item.value].mabtslr[4], res[item.value].mabtslr[5], res[item.value].mabtslr[6], res[item.value].mabtslr[7], res[item.value].mabtslr[8]],
              areaStyle: {
                  color: 'rgba(0, 158, 255, 0.15)' // 这个是关键
              },
              symbol: 'none', // 去掉点
            },
            {
              name: 'CMAB',
              value: [res[item.value].cmab[1], res[item.value].cmab[2], res[item.value].cmab[3], res[item.value].cmab[4], res[item.value].cmab[5], res[item.value].cmab[6], res[item.value].cmab[7], res[item.value].cmab[8]],
              areaStyle: {
                  color: 'rgba(255, 143, 0, 0.15)' // 这个是关键
              },
              symbol: 'none', // 去掉点
            },
          ]
          item.obj.indicatorArr = [
            {
              name: 'Level 1',
              max: res[item.value].total[1]
            },
            {
              name: 'Level 2',
              max: res[item.value].total[2]
            },
            {
              name: 'Level 3',
              max: res[item.value].total[3]
            },
            {
              name: 'Level 4',
              max: res[item.value].total[4]
            },
            {
              name: 'Level 5',
              max: res[item.value].total[5]
            },
            {
              name: 'Level 6',
              max: res[item.value].total[6]
            },
            {
              name: 'Level 7',
              max: res[item.value].total[7]
            },
            {
              name: 'Level 8',
              max: res[item.value].total[8]
            }
          ]
        } else {
          item.obj.arr = [
            {
              name: 'MAB',
              value: [res[item.value].mabtslr[1], res[item.value].mabtslr[2], res[item.value].mabtslr[3], res[item.value].mabtslr[4]],
              areaStyle: {
                  color: 'rgba(0, 158, 255, 0.15)' // 这个是关键
              },
              symbol: 'none', // 去掉点
            },
            {
              name: 'CMAB',
              value: [res[item.value].cmab[1], res[item.value].cmab[2], res[item.value].cmab[3], res[item.value].cmab[4]],
              areaStyle: {
                  color: 'rgba(255, 143, 0, 0.15)' // 这个是关键
              },
              symbol: 'none', // 去掉点
            },
          ]
          item.obj.indicatorArr = [
            {
              name: 'Level 1',
              max: res[item.value].total[1]
            },
            {
              name: 'Level 2',
              max: res[item.value].total[2]
            },
            {
              name: 'Level 3',
              max: res[item.value].total[3]
            },
            {
              name: 'Level 4',
              max: res[item.value].total[4]
            }
          ]
        }
      })
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
    min-height: 698px;
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
      // margin-top: 24px;
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
            .chart-icon{
              margin: 2px auto 0 4px;
              transform: rotate(180deg);
              color: #969696;
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
