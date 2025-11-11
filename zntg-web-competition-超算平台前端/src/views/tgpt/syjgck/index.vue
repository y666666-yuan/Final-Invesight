<template>
  <div class="syjgck">
   <ResultList v-if="!detailObj.algorithmName" @goNextStep="handleDetail" />
   <Product v-if="detailObj && detailObj.algorithmName" @goNextStep="handleDetail" :initData="detailObj" />
  </div>
</template>

<script setup>
import { cloneDeep } from "lodash-es";
import { ref, reactive, toRefs, onMounted, onBeforeUnmount, computed } from "vue";
import ResultList from "./components/resultList.vue";
import Product from "./components/product.vue";
const state = reactive({
  detailObj: {}
})
const { detailObj } = toRefs(state)
onMounted(() => {
 
});
const handleDetail = (obj) => {
  state.detailObj = obj
}
</script>

<style lang="scss" scoped>

</style>
