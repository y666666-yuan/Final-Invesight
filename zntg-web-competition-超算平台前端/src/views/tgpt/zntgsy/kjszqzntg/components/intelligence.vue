<template>
  <div class="intelligence">
    <div class="intelligence-main">
      <div class="intelligence-title">
        <div class="i-svg:title"></div>
        <div class="intelligence-name">智能投顾算法选择</div>
      </div>
      <div class="intelligence-con">
        <div class="con-name">
          <p>*</p>
          <p>智能投顾算法：</p>
          <div class="con-list">
            <div
              v-for="(item, index) in chackArr"
              :key="index"
              :class="['list-con', intelligenceActive === index ? 'list-con_active' : '']"
              @click="handleList(index)"
            >
              {{ item.name }}
            </div>
          </div>
        </div>
        <div class="intelligence-footer">
          <p @click="handleStep(3)">上一步</p>
          <p @click="handleStep(5)">下一步</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { debounce, cloneDeep, isPlainObject } from "lodash-es";
import { ref, reactive, toRefs, onMounted, onBeforeUnmount, computed } from "vue";
const emits = defineEmits(["goNextStep"]);
const props = defineProps({
  data: {
    default: [],
    type: Array,
  },
  initData: {
    default: [],
    type: Array,
  },
});
const intelligenceActive = ref(0);
const state = reactive({
  chackArr: cloneDeep(props.data),
});
const { chackArr } = toRefs(state);
onMounted(() => {
  state.chackArr.map((item) => {
    if (item.name === "强化学习") {
      item.property.map((ite) => {
        if (!isNotPlainObject(ite[ite.value])) {
          let arr = [];
          for (const key in ite[ite.value]) {
            arr.push({
              name: key,
              value: ite[ite.value][key],
            });
          }
          ite["defaultArr"] = arr;
        }
      });
    }
  });
});
const isNotPlainObject = (val) => {
  // 判断是否为对象
  return !isPlainObject(val);
};
const handleStrengthen = () => {
  // 添加新的参数
  state.chackArr[intelligenceActive.value].property.map((item) => {
    if (!isNotPlainObject(item[item.value])) {
      item.defaultArr.push({
        name: "",
        value: "",
      });
    }
  });
};
const handleList = (num) => {
  intelligenceActive.value = num;
  state.chackArr.map((item, index) => {
    if (index === num) {
      item.bol = true;
    } else {
      item.bol = false;
    }
  });
};
const closeSelect = () => {
  // 一键恢复参数
  state.chackArr[intelligenceActive.value] = cloneDeep(props.initData[intelligenceActive.value]);
  state.chackArr.map((item) => {
    if (item.name === "强化学习") {
      item.property.map((ite) => {
        if (!isNotPlainObject(ite[ite.value])) {
          let arr = [];
          for (const key in ite[ite.value]) {
            arr.push({
              name: key,
              value: ite[ite.value][key],
            });
          }
          ite["defaultArr"] = arr;
        }
      });
    }
  });
};
const handleStep = (num) => {
  if (num === 3) {
    emits("goNextStep", num, "up");
  } else {
    state.chackArr[intelligenceActive.value].bol = true;
    let chackArrBol = true;
    state.chackArr[intelligenceActive.value].property.map((item) => {
      if (isNotPlainObject(item[item.value])) {
        if (item[item.value] === "" || item[item.value] === null) {
          chackArrBol = false;
        }
      } else {
        item.defaultArr.map((ite) => {
          if (ite.name === "" || ite.name === null || ite.value === "" || ite.value === null) {
            chackArrBol = false;
          }
        });
      }
    });
    if (chackArrBol) {
      state.chackArr.map((item) => {
        if (item.name === "强化学习") {
          item.property.map((ite) => {
            if (!isNotPlainObject(ite[ite.value])) {
              let newObj = {};
              ite.defaultArr.map((it) => {
                newObj[it.name] = it.value - 0;
              });
              ite[ite.value] = newObj;
            }
          });
        }
      });
      emits("goNextStep", num, "down", state.chackArr);
    } else {
      ElMessage.warning("请填写完整数据！");
    }
  }
};
</script>

<style lang="scss" scoped>
.intelligence {
  display: flex;
  flex-direction: column;
  width: 100%;
  background: #f6f6f6;
  .intelligence-main {
    width: 100%;
    height: 698px;
    background: #ffffff;
    box-shadow: 0px 2px 4px 0px rgba(0, 0, 0, 0.06);
    .intelligence-title {
      display: flex;
      align-items: center;
      width: 100%;
      height: 45px;
      padding-left: 16px;
      border-bottom: 1px solid #eeeeee;
      color: #003d67;
      .intelligence-name {
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
    .intelligence-con {
      display: flex;
      flex-direction: column;
      width: 100%;
      margin-top: 24px;
      padding: 0 24px;
      .con-name {
        display: flex;
        align-items: center;
        height: 36px;
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
            width: 200px;
            height: 34px;
            background: #ffffff;
            border-radius: 2px 0px 0px 2px;
            font-family:
              PingFangSC,
              PingFang SC;
            font-weight: 400;
            font-size: 16px;
            color: rgba(0, 0, 0, 0.85);
            line-height: 34px;
            text-align: center;
            cursor: pointer;
          }
          .list-con_active {
            background: #eef8ff;
            color: #009eff;
          }
        }
      }
      .intelligence-line {
        display: flex;
        align-items: center;
        margin-top: 32px;
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
      .intelligence-checkbox {
        display: flex;
        flex-wrap: wrap;
        width: 100%;
        height: auto;
        padding-bottom: 26px;
        margin-top: 12px;
        background-color: #f4faff;
        .checkbox-con {
          display: flex;
          align-items: center;
          width: auto;
          height: auto;
          margin: 24px 132px 8px 0;
          .custom-checkbox {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            min-width: 158px;
            height: 30px;
            line-height: 30px;
            margin: 0;
            padding: 0;
            text-align: right;
            font-family:
              PingFangSC,
              PingFang SC;
            font-weight: 400;
            font-size: 16px;
            color: rgba(0, 0, 0, 0.85);
            p {
              margin: 0;
              padding: 0;
            }
            .custom-icon {
              height: 10px;
              margin-right: 5px;
              line-height: 18px;
              font-style: normal;
              color: #ff0000;
            }
          }
        }
      }
      .checkbox-con_active {
        flex-direction: column;
        height: auto;
        .checkbox-strengthen {
          align-items: flex-start;
        }
        .strengthen-main {
          display: flex;
          flex-direction: column;
          height: auto;
          justify-content: flex-start;
          .strengthen-con {
            display: flex;
            align-items: center;
            p {
              margin: 0 0 0 20px;
              padding: 0;
              color: #009eff;
              cursor: pointer;
            }
          }
          .strengthen-con:nth-of-type(n + 2) {
            margin-top: 20px;
          }
        }
      }
      .intelligence-footer {
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
