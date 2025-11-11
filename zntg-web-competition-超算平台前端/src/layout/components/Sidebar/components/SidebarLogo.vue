<template>
  <div class="logo">
    <transition enter-active-class="animate__animated animate__fadeInLeft">
      <router-link :key="+collapse" class="wh-full flex-center" to="/">
        <!-- <img :src="logo" class="w20px h20px" /> -->
        <!-- <span v-if="!collapse" class="title">
          {{ defaultSettings.title }}
        </span> -->
        <!-- <img src="@/assets/images/xncd-logo.png" alt="" />
        <img v-if="!collapse" src="@/assets/images/xncd-text.png" style="margin-left: 4px" alt="" /> -->
        <img src="@/assets/images/zt_logo.png" alt="" />
      </router-link>
    </transition>
  </div>
</template>

<script setup>
import defaultSettings from "@/settings";
import logo from "@/assets/logo.png";

defineProps({
  collapse: {
    type: Boolean,
    required: true,
  },
});
</script>

<style lang="scss" scoped>
.logo {
  width: 100%;
  // height: $navbar-height;
  height: 80px;
  // background-color: $sidebar-logo-background;
  background-color: #003d67;
  .title {
    flex-shrink: 0; /* 防止容器在空间不足时缩小 */
    margin-left: 10px;
    font-size: 14px;
    font-weight: bold;
    color: $sidebar-logo-text-color;
  }
}

.layout-top,
.layout-mix {
  .logo {
    width: $sidebar-width;
  }

  &.hideSidebar {
    .logo {
      width: $sidebar-width-collapsed;
    }
  }
}
</style>
