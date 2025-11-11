<template>
  <div :id="id" class="chart"></div>
</template>
<script setup>
import { reactive, ref, toRefs, watch, h, onMounted, nextTick } from "vue";
import * as echarts from "echarts";
import { cloneDeep } from "lodash-es";
const props = defineProps({
  id: {
    default: Math.random(),
    type: String,
  },
  chartData: {
    default: [],
    type: Array,
  },
});
watch(
  () => props.chartData,
  (newValue, oldValue) => {
    initChart();
  },
  { deep: true }
);
const initChart = () => {
  let colorList = ["#3BA0A5", "#3BA0FF", "#3BFFB4", "#00E5FF", "#FFD83B"];
  let colorAllList = [
    ["rgba(0, 124, 255, 0)", "rgba(0, 124, 255, 1)"],
    ["rgba(0, 229, 255, 0)", "rgba(0, 229, 255, 1)"],
    ["rgba(0, 193, 255, 0)", "rgba(0, 193, 255, 1)"],
    ["rgba(92, 255, 134, 0)", "rgba(92, 255, 134, 1)"],
    ["rgba(114, 211, 114, 0)", "rgba(114, 211, 114, 1)"],
    ["rgba(255, 178, 0, 0)", "rgba(255, 178, 0, 1)"],
    ["rgba(255, 96, 96, 0)", "rgba(255, 96, 96, 1)"],
    ["rgba(142, 143, 227, 0)", "rgba(142, 143, 227, 1)"],
    ["rgba(239, 124, 255, 0)", "rgba(239, 124, 255, 1)"],
  ];
  colorAllList.reverse();
  let xAxisData = [];
  let seriesData = [];
  let chartData = cloneDeep(props.chartData);
  chartData.forEach((item, index) => {
    xAxisData.push({
      value: item.label,
    });
    seriesData.push({
      value: item.score * 1000,
      itemStyle: {
        color: {
          type: "linear",
          x: item.score <= 0 ? 1 : 0,
          y: 0,
          x2: item.score <= 0 ? 0 : 1,
          y2: 0,
          colorStops: [
            {
              offset: 0,
              color: colorAllList[index][0],
            },
            {
              offset: 1,
              color: colorAllList[index][1],
            },
          ],
        },
      },
    });
  });
  let option = {
    grid: {
      show: true,
      left: 90,
      right: 20,
      top: 10,
      bottom: 20,
      backgroundColor: "rgba(255,255,255,0.05)",
      borderWidth: 0,
    },
    tooltip: {
      show: true,
      formatter: (params) => {
        let text = `${params.name}：${params.value}`;
        return text;
      },
    },
    xAxis: {
      type: "value",
    },
    yAxis: {
      type: "category",
      axisLabel: {
        fontSize: 14,
        color: "rgba(0,0,0,0.85)",
        lineHeight: 16,
        interval: 0,
        formatter: function (value, index) {
          let str = "";
          if (value?.length <= 4) {
            str = value;
          } else if (value?.length >= 5) {
            str = `${value.slice(0, 4)}\n${value.slice(4)}`;
          }
          return `{alignStyle|${str}}`;
        },
        rich: {
          alignStyle: {
            fontSize: 14,
            align: "right", // 左对齐文字，但不影响整体位置
          },
        },
      },
      axisTick: {
        show: false,
      },
      data: xAxisData,
    },
    series: [
      {
        barWidth: "16",
        itemStyle: {
          borderRadius: [2, 2, 0, 0],
        },
        data: seriesData,
        type: "bar",
      },
    ],
  };
  let chart = echarts.init(document.getElementById(props.id));
  chart.setOption(option);
};
onMounted(() => {
  nextTick(() => {
    initChart();
  });
});
</script>
<style lang="scss" scoped>
.chart {
  width: 100%;
  height: 100%;
}
</style>
