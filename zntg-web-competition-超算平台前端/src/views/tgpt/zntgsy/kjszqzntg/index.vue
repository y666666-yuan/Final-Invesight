<template>
  <div class="zntg">
    <div class="zntg-title">可解释增强的智能投顾</div>
    <div class="zntg-tab">
      <div
        v-for="(item, index) in tabArr"
        :key="index"
        :class="['tab-con', index < menuActive ? 'tab-con_active' : '']"
      >
        {{ item.name }}
      </div>
    </div>
    <div class="zntg-con" v-loading="zntgLoading" element-loading-text="结果计算中，请稍后">
      <ExperimentalData
        v-if="menuActive === 1"
        :data="experimentaObj"
        @goNextStep="handleStep"
        @handleDataset="handleDataset"
      />
      <Investor
        v-if="menuActive === 2 && investorData.length > 0"
        :data="investorData"
        @goNextStep="handleStep"
      />
      <Financial
        v-if="menuActive === 3 && financialData.length > 0"
        :data="financialData"
        @goNextStep="handleStep"
      />
      <Intelligence
        v-if="menuActive === 4 && intelligenceData.length > 0"
        :data="intelligenceData"
        :initData="initIntelligenceData"
        @goNextStep="handleStep"
      />
      <!-- <Product
        v-if="menuActive === 5"
        :taskId="taskId"
        :algorithmType="algorithmType"
        @goNextStep="handleStep"
      /> -->
      <Evaluate v-if="menuActive === 5" :taskId="taskId" :algorithmType="algorithmType" :algorithmName="algorithmName" @goNextStep="handleStep" />
    </div>
  </div>
</template>

<script setup>
import { cloneDeep } from "lodash-es";
import { ref, reactive, toRefs, onMounted, onBeforeUnmount, computed } from "vue";
import ExperimentalData from "./components/experimentalData.vue";
import Investor from "./components/investor.vue";
import Financial from "./components/financial.vue";
import Intelligence from "./components/intelligence.vue";
import Product from "./components/product.vue";
import Evaluate from "./components/evaluate.vue";
import ZntgsyAPI from "@/api/zntgsy.api";
const menuActive = ref(1);
const state = reactive({
  zntgLoading: false,
  tabArr: [
    {
      name: "1.实验数据选择",
      bol: true,
    },
    {
      name: "2.投资者特征选择",
      bol: false,
    },
    {
      name: "3.金融产品特征选择",
      bol: false,
    },
    {
      name: "4.智能投顾算法选择",
      bol: false,
    },
    // {
    //   name: "5.金融产品推荐结果",
    //   bol: false,
    // },
    {
      name: "5.推荐结果及解释",
      bol: false,
    },
  ],
  experimentaObj: {}, // 实验数据初始值
  investorData: [],
  financialData: [],
  intelligenceData: [],
  initIntelligenceData: [],
  taskId: "",
  algorithmType: "",
  algorithmName: "",
});
const {
  zntgLoading,
  tabArr,
  experimentaObj,
  investorData,
  financialData,
  intelligenceData,
  initIntelligenceData,
  taskId,
  algorithmType,
  algorithmName,
} = toRefs(state);
onMounted(() => {
  getAlgorithmList();
});
const getQueryFeatures = (name) => {
  // 获取实验对象属性
  ZntgsyAPI.getQueryFeatures({ dataset_name: name }).then((data) => {
    if (data.investor_features) {
      // 投资者特征
      const arr = Object.entries(data.investor_features).map(([key, value]) => ({ key, value }));
      arr.map((item) => {
        state.investorData.push({
          title: item.value,
          value: item.key,
          bol: true,
        });
      });
    }
    if (data.product_features) {
      // 金融产品特征
      const arr = Object.entries(data.product_features).map(([key, value]) => ({ key, value }));
      arr.map((item) => {
        state.financialData.push({
          title: item.value,
          value: item.key,
          bol: true,
        });
      });
    }
  });
};
const getAlgorithmList = () => {
  // 获取已有算法
  ZntgsyAPI.getAlgorithmList({ algorithm_type: 4 }).then((data) => {
    if (data.length > 0) {
      data.map((item) => {
        item.bol = false;
        if (item.property.length > 0) {
          item.property.map((ite) => {
            ite[ite.value] = ite.default_value;
          });
        }
      });
      state.intelligenceData = data;
      state.initIntelligenceData = data;
    }
  });
};
const handleDataset = (val) => {
  // 实验数据改变
  if (val) {
    state.investorData = [];
    state.financialData = [];
    getQueryFeatures(val);
  }
};
const postRunTask = () => {
  // 执行算法任务
  state.zntgLoading = true;
  let params = cloneDeep(state.experimentaObj);
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
  let intelligenceInfo = state.intelligenceData.filter((item) => {
    return item.bol;
  })[0];
  let algorithmObj = {};
  intelligenceInfo.property.map((item) => {
    algorithmObj[item.value] = item[item.value];
  });
  state.algorithmName = intelligenceInfo.name;
  let newParams = {
    dataset_name: params.dataset_name,
    filter: params.filter,
    algorithm_type: 4,
    algorithm_name: intelligenceInfo.value,
    algorithm_params: algorithmObj,
    investor_features: state.investorData
      .filter((item) => {
        return item.bol;
      })
      .map((item) => item.value),
    product_features: state.financialData
      .filter((item) => {
        return item.bol;
      })
      .map((item) => item.value),
  };

  ZntgsyAPI.postRunTask(newParams).then((data) => {
    state.taskId = data;
    state.algorithmType = newParams.algorithm_name;
    state.zntgLoading = false;
    menuActive.value = 5;
  }).catch((error) => {
    state.zntgLoading = false;
  });
};
const handleStep = (num, type, obj) => {
  if (!(num === 5 && type === 'down')) {
    menuActive.value = num;
  }
  if (type === "down") {
    if (num === 1) {
      state.experimentaObj = {};
    } else if (num === 2) {
      state.experimentaObj = cloneDeep(obj);
    } else if (num === 3) {
      state.investorData = obj;
    } else if (num === 4) {
      state.financialData = obj;
    } else if (num === 5) {
      state.intelligenceData = obj;
      postRunTask();
    }
  }
};
</script>

<style lang="scss" scoped>
.zntg {
  padding: 0 16px;
  background: #f6f6f6;
  .zntg-title {
    height: 25px;
    font-family:
      PingFangSC,
      PingFang SC;
    font-weight: 600;
    font-size: 18px;
    color: #003d67;
    line-height: 25px;
  }
  .zntg-tab {
    display: flex;
    align-items: center;
    width: 1648px;
    height: 56px;
    margin-top: 15px;
    .tab-con {
      display: flex;
      justify-content: center;
      align-items: center;
      width: 320px;
      height: 56px;
      background: url("@/assets/images/step3.png") center center no-repeat;
      background-size: 100% 100%;
      font-family:
        PingFangSC,
        PingFang SC;
      font-weight: 400;
      font-size: 16px;
      color: #003d67;
      line-height: 22px;
      &.tab-con_active {
        background: url("@/assets/images/step4.png") center center no-repeat;
        background-size: 100% 100%;
      }
    }
    .tab-con:nth-of-type(1) {
      background: url("@/assets/images/step2.png") center center no-repeat;
      background-size: 100% 100%;
      &.tab-con_active {
        background: url("@/assets/images/step1.png") center center no-repeat;
        background-size: 100% 100%;
      }
    }
    .tab-con:nth-last-of-type(1) {
      background: url("@/assets/images/step6.png") center center no-repeat;
      background-size: 100% 100%;
      &.tab-con_active {
        background: url("@/assets/images/step5.png") center center no-repeat;
        background-size: 100% 100%;
      }
    }
    .tab-con:nth-of-type(n + 2) {
      margin-left: 12px;
    }
    .tab-con_active {
      font-weight: 600;
      color: #ffffff;
    }
  }
  .zntg-con {
    display: flex;
    flex-direction: column;
    margin-top: 16px;
    padding-bottom: 16px;
  }
}
</style>
