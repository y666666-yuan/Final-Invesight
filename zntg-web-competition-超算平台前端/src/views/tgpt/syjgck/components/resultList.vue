<template>
  <div class="product">
    <div class="product-main">
      <div class="product-title">
        <div class="product-name">测试结果列表</div>
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
                  @click="handleClick(scope.row)"
                >
                  详情
                </p>
              </template>
            </el-table-column>
          </el-table>
          <div v-if="productTotal > 0" class="checkbox-page">
            <el-pagination
              v-model:current-page="productObj.current"
              v-model:page-size="productObj.size"
              layout="total,prev, pager, next, jumper"
              :total="productTotal"
              @current-change="handleCurrentChange"
            />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { debounce, cloneDeep } from "lodash-es";
import { ref, reactive, toRefs, onMounted, onBeforeUnmount, computed } from "vue";
import Bar from "./bar/bar.vue";
import ZntgsyAPI from "@/api/zntgsy.api";
import SyjgckAPI from "@/api/syjgck.api";
const emits = defineEmits(["goNextStep"]);
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
      label: "测试名称",
      prop: "expName",
    },
    {
      label: "算法名称",
      prop: "algorithmName",
    },
    {
      label: "测试类型",
      prop: "expTypeName",
    },
    {
      label: "测试时间",
      prop: "createdAt",
    },
    {
      label: "操作",
      prop: "action",
      width: 200,
    },
  ],
  productData: [],
  productObj: {
    current: 1,
    size: 5,
  },
  productTotal: 0,
});
const { loading, productColumns, productData, productObj, productTotal } =
  toRefs(state);
onMounted(() => {
  postPageExp();
});
const postPageExp = () => {
  // 获取测试结果
  state.loading = true;
  state.productData = [];
  SyjgckAPI.postPageExp({ ...state.productObj }).then((data) => {
    if (data.records.length > 0) {
      data.records.map((item, inde) => {
        item.index = (state.productObj.current - 1) * state.productObj.size + inde + 1;
      });
      state.productData = data.records;
      state.productTotal = data.total;
      state.loading = false;
    } else {
      state.loading = false;
    }
  });
};
const handleCurrentChange = (val) => {
  state.productObj.current = val;
  postPageExp();
};
const handleClick = (obj) => {
  emits("goNextStep", obj);
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
    .product-title {
      display: flex;
      // align-items: center;
      width: 100%;
      height: 40px;
      padding-left: 16px;
      color: #003d67;
      .product-name {
        height: 22px;
        font-family:
          PingFangSC,
          PingFang SC;
        font-weight: 600;
        font-size: 18px;
        line-height: 22px;
      }
    }
    .product-con {
      display: flex;
      flex-direction: column;
      width: calc(100% - 32px);
      min-height: 698px;
      margin: 0 16px;
      padding: 24px;
      background-color: #ffffff;
      box-shadow: 0px 2px 4px 0px rgba(0,0,0,0.06);
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
          max-height: 354px;
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
