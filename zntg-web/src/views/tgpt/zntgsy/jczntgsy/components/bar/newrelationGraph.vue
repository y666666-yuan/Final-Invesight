<template>
  <div
    ref="wrapperRef"
    class="chart-wrapper"
    @mouseenter="disableScroll"
    @mouseleave="enableScroll"
  >
    <div ref="chartRef" class="chart"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from "vue";
import * as echarts from "echarts";
import { cloneDeep } from "lodash-es";

const props = defineProps({
  chartData: {
    type: Object,
    default: () => ({}),
  },
});

const wrapperRef = ref(null);
const chartRef = ref(null);
let chart = null;
let scrollContainer = null;
let originalOverflow = "";

function getScrollableParent(el) {
  while (el && el !== document.body) {
    const style = window.getComputedStyle(el);
    const overflowY = style.overflowY;
    if (overflowY === "auto" || overflowY === "scroll") return el;
    el = el.parentElement;
  }
  return document.body;
}

function disableScroll() {
  if (scrollContainer) {
    originalOverflow = scrollContainer.style.overflow;
    scrollContainer.style.overflow = "hidden";
  }
}

function enableScroll() {
  if (scrollContainer) {
    scrollContainer.style.overflow = originalOverflow || "";
  }
}

function initChart() {
  const el = chartRef.value;
  if (!el || !props.chartData?.nodes || !props.chartData?.edges) return;
  const tzArr = props.chartData.nodes.filter(item => item.label === '投资者');
  const totalNumbers = tzArr.map(item => item.properties.total_bid_number);
  let maxTz = 0;
  let minTz = 0;
  if (totalNumbers.length > 0) {
    maxTz = Math.max(...totalNumbers);
    minTz = Math.min(...totalNumbers);
  }
  const cpArr = props.chartData.nodes.filter(item => item.label === '产品');
  const cpTotalNumbers = cpArr.map(item => item.properties.project_invest_count);
  let maxCp = 0;
  let minCp = 0;
  if (cpTotalNumbers.length > 0) {
    maxCp = Math.max(...cpTotalNumbers);
    minCp = Math.min(...cpTotalNumbers);
  }
  const nodes = props.chartData.nodes.map((item) => ({
    ...item,
    symbolSize: 12 + (item.label === "投资者" ? ((item.properties.total_bid_number - minTz) / (maxTz - minTz)) * 48 : ((item.properties.project_invest_count - minCp) / (maxCp - minCp)) * 48),
    category: item.label === "投资者" ? 0 : 1,
  }));
  console.log(nodes);

  const edges = props.chartData.edges.map((item) => ({
    source: item.edge_to,
    target: item.edge_from,
    tooltip: { show: false },
  }));
  if (chart) {
    chart.dispose();
  }
  chart = echarts.init(el);
  chart.setOption({
    color: ["#009EFF", "#60D966", "#3BFFB4", "#00E5FF", "#FFD83B"],
    tooltip: {
      trigger: 'item',
      formatter: function (params) {
        if (params.data.label === '产品') {
          return `
            <div style="
                  display: flex;
                  flex-direction: column;
                  width: 400px;
                  height: 208px;
                  padding: 4px;
                  background: #FFFFFF;
                  box-shadow: 0px 2px 4px 0px rgba(0,0,0,0.1);
                  border-radius: 2px;
                  border: 1px solid #E8E8E8;">
              <div style="
                          display: flex;
                          align-items: center;
                          width: 392px;
                          height: 36px;
                          padding-left: 12px;
                          background: #DAF7DA;
                          border-radius: 2px;
                          font-family: PingFangSC, PingFang SC;
                          font-weight: 400;
                          font-size: 16px;
                          color: #146E00;">产品编号：${params.data.id.replace('p_', '')}</div>
              <div style="
                  display: flex;
                  flex-wrap: wrap;
                  align-content: flex-start;
                  width: 392px;
                  height: 164px;
                  font-family: PingFangSC, PingFang SC;
                  font-weight: 400;
                  font-size: 16px;
                  color: rgba(0,0,0,0.65);">
                <p style="width: 206px;height: 22px;padding: 0 0 0 12px;margin: 16px 0 0 0;">风险等级：${params.data.properties.level}</p>
                <p style="width: 186px;height: 22px;padding: 0 0 0 12px;margin: 16px 0 0 0;">利率：${params.data.properties.apr_percent}</p>
                <p style="width: 206px;height: 22px;padding: 0 0 0 12px;margin: 16px 0 0 0;">期限：${params.data.properties.term}个月</p>
                <p style="width: 186px;height: 22px;padding: 0 0 0 12px;margin: 16px 0 0 0;">规模：${params.data.properties.total}</p>
                <p style="width: 206px;height: 22px;padding: 0 0 0 12px;margin: 16px 0 0 0;overflow: hidden;white-space: nowrap;text-overflow: ellipsis;">还款方式：${params.data.properties.repayment}</p>
                <p style="width: 186px;height: 22px;padding: 0 0 0 12px;margin: 16px 0 0 0;">被投资人数：${params.data.properties.project_invest_count}</p>
              </div>
            </div>
          `;
        } else {
          return `
            <div style="
                  display: flex;
                  flex-direction: column;
                  width: 400px;
                  height: 208px;
                  padding: 4px;
                  background: #FFFFFF;
                  box-shadow: 0px 2px 4px 0px rgba(0,0,0,0.1);
                  border-radius: 2px;
                  border: 1px solid #E8E8E8;">
              <div style="
                          display: flex;
                          align-items: center;
                          width: 392px;
                          height: 36px;
                          padding-left: 12px;
                          background: #E2F4FF;
                          border-radius: 2px;
                          font-family: PingFangSC, PingFang SC;
                          font-weight: 400;
                          font-size: 16px;
                          color: #003E6E;">投资者编号：${params.data.id.replace('u_', '')}</div>
              <div style="
                  display: flex;
                  flex-wrap: wrap;
                  align-content: flex-start;
                  width: 392px;
                  height: 164px;
                  font-family: PingFangSC, PingFang SC;
                  font-weight: 400;
                  font-size: 16px;
                  color: rgba(0,0,0,0.65);">
                <p style="width: 206px;height: 22px;padding: 0 0 0 12px;margin: 16px 0 0 0;">性别：${params.data.properties.gender}</p>
                <p style="width: 186px;height: 22px;padding: 0 0 0 12px;margin: 16px 0 0 0;">教育背景：${params.data.properties.edu_level}</p>
                <p style="width: 206px;height: 22px;padding: 0 0 0 12px;margin: 16px 0 0 0;">年龄：${params.data.properties.age}</p>
                <p style="width: 186px;height: 22px;padding: 0 0 0 12px;margin: 16px 0 0 0;">毕业院校：${params.data.properties.university}</p>
                <p style="width: 206px;height: 22px;padding: 0 0 0 12px;margin: 16px 0 0 0;">身份：${params.data.properties.current_identity}</p>
                <p style="width: 186px;height: 22px;padding: 0 0 0 12px;margin: 16px 0 0 0;">投资次数：${params.data.properties.total_bid_number}</p>
              </div>
            </div>
          `;
        }
      },
      backgroundColor: 'transparent', // 背景透明，由 HTML 控制背景
      borderWidth: 0,                 // 去掉默认边框
      extraCssText: 'box-shadow:none;' // 可选：去除阴影
    },
    legend: { show: false },
    series: [
      {
        type: "graph",
        layout: "force",
        center: ['50%', '50%'],
        data: nodes,
        links: edges,
        categories: [{ name: "投资者" }, { name: "产品" }],
        roam: true,
        draggable: false,
        label: { position: "right" },
        force: {
          initLayout: null, // 重要：重新计算布局
          repulsion: 10000,
          edgeLength: [50, 200],
          gravity: 0.2,
          layoutAnimation: false,
        },
        animation: true,
        lineStyle: {
          color: "rgba(212,212,212,0.5)",
        },
        emphasis: {
          focus: 'adjacency',
          lineStyle: {
            width: 5
          }
        }
      },
    ],
  }, true);
}
watch(
  () => props.chartData,
  () => {
    initChart();
  },
  { deep: true }
);
onMounted(() => {
  scrollContainer = getScrollableParent(wrapperRef.value);
  initChart();
});

onBeforeUnmount(() => {
  if (chart) chart.dispose();
  enableScroll();
});
</script>

<style scoped>
.chart-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 800px;
  overflow: hidden;
  touch-action: none;
}
.chart {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 800px;
}
</style>
