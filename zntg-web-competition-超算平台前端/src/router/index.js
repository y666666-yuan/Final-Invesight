import { createRouter, createWebHashHistory } from "vue-router";

export const Layout = () => import("@/layout/index.vue");

// 静态路由
export const constantRoutes = [
  {
    path: "/redirect",
    component: Layout,
    meta: { hidden: true },
    children: [
      {
        path: "/redirect/:path(.*)",
        component: () => import("@/views/redirect/index.vue"),
      },
    ],
  },

  {
    path: "/login",
    component: () => import("@/views/login/index.vue"),
    meta: { hidden: true },
  },

  {
    path: "/dashboard",
    component: Layout,
    redirect: "/dashboard",
    children: [
      {
        path: "dashboard",
        component: () => import("@/views/dashboard/index.vue"),
        // 用于 keep-alive 功能，需要与 SFC 中自动推导或显式声明的组件名称一致
        // 参考文档: https://cn.vuejs.org/guide/built-ins/keep-alive.html#include-exclude
        name: "Dashboard",
        meta: {
          title: "dashboard",
          icon: "homepage",
          affix: true,
          keepAlive: true,
        },
      },
      {
        path: "401",
        component: () => import("@/views/error/401.vue"),
        meta: { hidden: true },
      },
      {
        path: "404",
        component: () => import("@/views/error/404.vue"),
        meta: { hidden: true },
      },
      {
        path: "profile",
        name: "Profile",
        component: () => import("@/views/profile/index.vue"),
        meta: { title: "个人中心", icon: "user", hidden: true },
      },
      {
        path: "my-notice",
        name: "MyNotice",
        component: () => import("@/views/system/notice/components/MyNotice.vue"),
        meta: { title: "我的通知", icon: "user", hidden: true },
      },
    ],
  },

  {
    path: "/",
    redirect: "/jczntgsy",
    component: Layout,
    meta: {
      title: "智能投顾沙盒",
      icon: "sy",
    },
    children: [
      {
        name: "jczntgsy",
        path: "/jczntgsy",
        meta: {
          title: "导航式开发测试平台",
          icon: "",
        },
        component: () => import("@/views/tgpt/zntgsy/jczntgsy/index.vue"),
      },
      {
        name: "ystjzntg",
        path: "/ystjzntg",
        meta: {
          title: "带约束条件的智能投顾",
          icon: "",
        },
        component: () => import("@/views/tgpt/zntgsy/ystjzntg/index.vue"),
      },
      {
        name: "xtzzzntg",
        path: "/xtzzzntg",
        meta: {
          title: "对新投资者的智能投顾",
          icon: "",
        },
        component: () => import("@/views/tgpt/zntgsy/xtzzzntg/index.vue"),
      },
      {
        name: "kjszqzntg",
        path: "/kjszqzntg",
        meta: {
          title: "可解释增强的智能投顾",
          icon: "",
        },
        component: () => import("@/views/tgpt/zntgsy/kjszqzntg/index.vue"),
      },
    ],
  },
  {
    path: "/syjg",
    redirect: "/syjgck",
    component: Layout,
    meta: {
      title: "测试结果查看",
      icon: "syjgck",
    },
    children: [
      {
        name: "syjgck",
        path: "/syjgck",
        meta: {
          title: "测试结果查看",
          icon: "syjgck",
        },
        component: () => import("@/views/tgpt/syjgck/index.vue"),
      },
    ],
  },
  // {
  //   path: "/sybg",
  //   redirect: "/sybgzx",
  //   component: Layout,
  //   meta: {
  //     title: "实验报告撰写",
  //     icon: "sybg",
  //   },
  //   children: [
  //     {
  //       name: "sybgzx",
  //       path: "/sybgzx",
  //       meta: {
  //         title: "实验报告撰写",
  //         icon: "sybg",
  //       },
  //       component: () => import("@/views/tgpt/sybgzx/index.vue"),
  //     },
  //   ],
  // },
  // {
  //   path: "/zn",
  //   redirect: "/znzj",
  //   component: Layout,
  //   meta: {
  //     title: "智能助教",
  //     icon: "znzj",
  //   },
  //   children: [
  //     {
  //       name: "znzj",
  //       path: "/znzj",
  //       meta: {
  //         title: "智能助教",
  //         icon: "znzj",
  //       },
  //       component: () => import("@/views/tgpt/znzj/index.vue"),
  //     },
  //   ],
  // },
];

/**
 * 创建路由
 */
const router = createRouter({
  history: createWebHashHistory(),
  routes: constantRoutes,
  // 刷新时，滚动条位置还原
  scrollBehavior: () => ({ left: 0, top: 0 }),
});

// 全局注册 router
export function setupRouter(app) {
  app.use(router);
}

export default router;
