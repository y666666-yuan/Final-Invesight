<template>
  <div class="navbar-top">
    <img src="@/assets/images/zt_text.png" alt="" />
    <NavbarRight />
  </div>
  <div class="navbar">
    <div class="navbar__left">
      <!-- 展开/收缩菜单  -->
      <Hamburger :is-active="isSidebarOpened" @toggle-click="toggleSideBar" />
      <!-- 面包屑 -->
      <Breadcrumb />
    </div>
    <!-- 导航栏右侧 -->
    <!-- <NavbarRight /> -->
  </div>
</template>

<script setup>
import { useAppStore } from "@/store";

const appStore = useAppStore();

// 侧边栏是否打开
const isSidebarOpened = computed(() => appStore.sidebar.opened);

// 展开/收缩菜单
function toggleSideBar() {
  appStore.toggleSidebar();
}
</script>

<style lang="scss" scoped>
.navbar {
  display: flex;
  justify-content: space-between;
  height: $navbar-height;
  // background: var(--el-bg-color);
  background: #f6f6f6;

  &__left {
    display: flex;
    align-items: center;
  }
}
.navbar-top {
  height: 80px;
  background: #02538a;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-left: 24px;
  color: #fff;
}
</style>
