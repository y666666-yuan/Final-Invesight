<template>
  <div class="financial">
    <div class="financial-main">
      <div class="financial-title">
        <div class="i-svg:title"></div>
        <div class="financial-name">金融产品特征选择</div>
      </div>
      <div class="financial-con">
        <div class="con-name">
          <p>*</p>
          <p>金融产品特征：</p>
          <p @click="closeSelect">清空筛选</p>
        </div>
        <div class="financial-checkbox">
          <div
            v-for="(item, index) in chackArr"
            :key="index"
            class="checkbox-con"
            :class="{ 'checkbox-checked': item.bol }"
            @click="handleCheck(index)"
          >
            <p class="custom-checkbox"></p>
            <p class="label-text">{{ item.title }}</p>
          </div>
        </div>
        <div class="financial-footer">
          <p @click="handleStep(2)">上一步</p>
          <p @click="handleStep(4)">下一步</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { debounce, cloneDeep } from "lodash-es";
import { ref, reactive, toRefs, onMounted, onBeforeUnmount, computed } from "vue";
const emits = defineEmits(["goNextStep"]);
const props = defineProps({
  data: {
    default: [],
    type: Array,
  },
});
const state = reactive({
  chackArr: cloneDeep(props.data),
  checkedCities: [],
});
const { chackArr, checkedCities } = toRefs(state);
onMounted(() => {
  state.checkedCities = state.chackArr.filter((item) => item.bol).map((item) => item.value);
});
const handleCheck = (num) => {
  state.chackArr[num].bol = !state.chackArr[num].bol;
  state.checkedCities = state.chackArr.filter((item) => item.bol).map((item) => item.value);
};
const handleStep = (num) => {
  if (num == 2) {
    emits("goNextStep", num, "up");
  } else {
    if (state.checkedCities.length > 0) {
      emits("goNextStep", num, "down", state.chackArr);
    } else {
      ElMessage.warning("请至少选择一类特征！");
    }
  }
};
const closeSelect = () => {
  // 清空筛选
  state.chackArr.map((item) => {
    item.bol = false;
  });
  state.checkedCities = [];
};
</script>

<style lang="scss" scoped>
.financial {
  display: flex;
  flex-direction: column;
  width: 100%;
  background: #f6f6f6;
  .financial-main {
    width: 100%;
    height: 698px;
    background: #ffffff;
    box-shadow: 0px 2px 4px 0px rgba(0, 0, 0, 0.06);
    .financial-title {
      display: flex;
      align-items: center;
      width: 100%;
      height: 45px;
      padding-left: 16px;
      border-bottom: 1px solid #eeeeee;
      color: #003d67;
      .financial-name {
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
    .financial-con {
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
      .financial-checkbox {
        display: flex;
        flex-wrap: wrap;
        width: 1348px;
        height: auto;
        margin-top: 16px;
        .checkbox-con {
          display: flex;
          align-items: center;
          width: 252px;
          height: 56px;
          margin: 0 16px 20px 0;
          padding-left: 24px;
          background: #f4faff;
          border-radius: 2px;
          border: 1px solid #dff0ff;
          cursor: pointer;
          &.checkbox-checked {
            background: #009eff;
            border: 1px solid transparent;
          }
          .custom-checkbox {
            width: 16px;
            height: 16px;
            background: #ffffff;
            border-radius: 2px;
            border: 1px solid #e1e1e1;
            margin-right: 8px;
            position: relative;
          }
          .label-text {
            height: 22px;
            margin: -2px 0 0 0;
            font-family:
              PingFangSC,
              PingFang SC;
            font-weight: 400;
            font-size: 16px;
            color: rgba(0, 0, 0, 0.85);
            line-height: 22px;
          }

          &.checkbox-checked .custom-checkbox::after {
            content: "";
            position: absolute;
            left: 4px;
            top: 0px;
            width: 6px;
            height: 11px;
            border: solid #409eff;
            border-width: 0 2px 2px 0;
            transform: rotate(45deg);
          }
          &.checkbox-checked .label-text {
            color: #ffffff;
          }
        }
      }
      .financial-footer {
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
</style>
