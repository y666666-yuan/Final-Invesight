<template>
  <div :id="id" class="chart"></div>
</template>
<script setup>
import { reactive, ref, toRefs, watch, h, onMounted, nextTick } from 'vue';
import * as echarts from 'echarts';
import { debounce, cloneDeep } from "lodash-es";
const props = defineProps({
  id: {
      default: Math.random(),
      type: String,
  },
  type: {
    default: 0,
    type: Number
  },
  chartObj: {
      default: {},
      type: Object,
  }
})
watch(
  () => [props.type, props.chartObj],
  ([newValue, newValueB], [oldValue, oldValueB]) => {
      initChart()
  },
  { deep: true }
)
const initChart = () => {
  let colorList = ['rgba(0, 158, 255, 1)','rgba(255, 128, 0, 1)','#3BFFB4','#00E5FF','#FFD83B'];
  let chartObj = cloneDeep(props.chartObj)
  let option = {
    color: colorList,
    tooltip: {
      show:true,
      backgroundColor: 'rgba(0, 0, 0, .9)',
      borderWidth: 0,
      textStyle: {
        color: 'rgba(255, 255, 255, .9)'
      },
    },
    legend: {
      data: ['MAB', 'CMAB'],
      textStyle: {
        color: 'rgba(0, 0, 0, 0.85)',  // 图例文字颜色
        fontSize: 12
      },
      itemWidth: 12,
      itemHeight: 6,
      top: 20,
      right: 16
    },
    // grid: {
    //     top: 0,
    //     bottom: 50,
    //     left: 30,
    //     right: 30,
    // },
    radar: {
      // shape: 'circle',
      radius:'70%',
      nameGap: 5,
      center: ['50%', '50%'],
      name: {
        // formatter: function (name) {
        //   // 例如：每 2 个字符换一行
        //   return name.length > 2 ? name.slice(0, 2) + '\n' + name.slice(2) : name;
        // },
        textStyle: {
          lineHeight: 16,
          color: '#8FA5D7'
        }
      },
      indicator: chartObj.indicatorArr,
      splitArea: {
        show: true,
        areaStyle: {
          color: [
            'rgba(248,248,248,1)',
            'rgba(248,248,248,1)',
          ], // 每一层的背景色
        }
      },
      splitLine: {
        lineStyle: {
          width: 2,
          color: 'rgba(238, 238, 238, 1)', // 分割线颜色
        }
      },
      axisLine: {
        lineStyle: {
          width: 2,
          color: 'rgba(238, 238, 238, 1)', // 轴线颜色
        }
      }
    },
    series: [
      {
        name: 'Budget vs spending',
        type: 'radar',
        data: chartObj.arr,
        // label:{
        //   show:true,
        //   formatter: function (value) {
        //     return value.data.name; // 保留两位小数
        //   }
        // }
      },
    ]
  };
  let chart = echarts.init(document.getElementById(props.id));
  chart.setOption(option)
}
onMounted(() => {
  nextTick(()=>{
    initChart()
  })
})
</script>
<style scoped lang="scss">
.chart {
  width: 100%;
  height: 100%;
}
</style>
