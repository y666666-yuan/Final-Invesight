<template>
  <div ref="wrapperRef" class="chart-wrapper">
    <div :id="id" ref="chartContainerRef" class="chart"></div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick, onMounted, onBeforeUnmount } from "vue";
import * as echarts from "echarts";
import { cloneDeep } from "lodash-es";

const props = defineProps({
  id: {
    default: () => "chart-" + Math.random().toString(36).slice(2),
    type: String,
  },
  chartData: {
    default: () => ({}),
    type: Object,
  },
});

const wrapperRef = ref(null);
const chartContainerRef = ref(null);
let chart = null;
let preventScrollHandlers = [];

watch(
  () => props.chartData,
  () => {
    initChart();
  },
  { deep: true }
);

const initChart = () => {
  const chartData = cloneDeep(props.chartData);
  if (!chartData?.nodes || !chartData?.edges) return;

  const newNodes = chartData.nodes.map((item) => ({
    ...item,
    symbolSize: 12,
    category: item.label === "投资者" ? 0 : 1,
  }));

  const newLinks = chartData.edges.map((item) => ({
    source: item.edge_to,
    target: item.edge_from,
    tooltip: { show: false },
  }));

  const option = {
    color: ["#009EFF", "#60D966", "#3BFFB4", "#00E5FF", "#FFD83B"],
    tooltip: {},
    legend: { show: false },
    series: [
      {
        type: "graph",
        layout: "force",
        data: newNodes,
        links: newLinks,
        categories: [{ name: "投资者" }, { name: "产品" }],
        roam: true,
        label: { position: "right" },
        force: {
          repulsion: 2000,
          edgeLength: [50, 200],
          gravity: 0.05,
          layoutAnimation: false,
        },
        animation: false,
        lineStyle: {
          color: "rgb(212,212,212,0.5)",
        },
      },
    ],
  };

  chart = echarts.init(chartContainerRef.value);
  chart.setOption(option);
};

onMounted(() => {
  nextTick(() => {
    initChart();
    blockDefaultScroll();
  });
});

onBeforeUnmount(() => {
  if (chart) chart.dispose();
  cleanupScrollBlocking();
});

// 核心：阻止默认滚动行为，避免 passive 报错
function blockDefaultScroll() {
  const el = chartContainerRef.value;
  if (!el) return;

  const prevent = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const options = { passive: false };

  // 放在 chartContainerRef 上，阻止滚轮和触摸滚动
  el.addEventListener("wheel", prevent, options);
  el.addEventListener("touchmove", prevent, options);

  preventScrollHandlers = [
    { type: "wheel", handler: prevent },
    { type: "touchmove", handler: prevent },
  ];
}

function cleanupScrollBlocking() {
  const el = chartContainerRef.value;
  if (!el) return;
  preventScrollHandlers.forEach(({ type, handler }) => {
    el.removeEventListener(type, handler);
  });
  preventScrollHandlers = [];
}
</script>

<style scoped>
.chart-wrapper {
  width: 1000px;
  height: 800px;
  overflow: hidden;
  overscroll-behavior: none;
  touch-action: none;
}
.chart {
  width: 100%;
  height: 100%;
}
</style>
