<template>
  <div class="login">
    <div class="login-img">
      <img src="@/assets/images/zti.png" alt="" />
    </div>
    <div class="login-con">
      <p class="login-title">欢迎登录</p>
      <div class="login-name">
        <img src="@/assets/images/user.png" alt="" />
        <el-input v-model="loginFormData.username" placeholder="请输入账号" />
      </div>
      <div class="login-pass">
        <img src="@/assets/images/password.png" alt="" />
        <el-input v-model="loginFormData.password" type="password" placeholder="请输入密码" />
      </div>
      <p class="login-btn" @click="handleLoginSubmit">登录</p>
    </div>
  </div>
</template>

<script setup>
import { useRoute } from "vue-router";
import { useI18n } from "vue-i18n";

import AuthAPI from "@/api/auth.api";
import router from "@/router";

import defaultSettings from "@/settings";
import { ThemeMode } from "@/enums/settings/theme.enum";

import { useSettingsStore, useUserStore } from "@/store";

const userStore = useUserStore();
const settingsStore = useSettingsStore();

const route = useRoute();
const { t } = useI18n();
const loginFormRef = ref();

const isDark = ref(settingsStore.theme === ThemeMode.DARK); // 是否暗黑模式
const loading = ref(false); // 按钮 loading 状态
const isCapsLock = ref(false); // 是否大写锁定
const captchaBase64 = ref(); // 验证码图片Base64字符串

const loginFormData = ref({
  username: "admin",
  password: "admin123",
});

const loginRules = computed(() => {
  return {
    username: [
      {
        required: true,
        trigger: "change",
        message: t("login.message.username.required"),
      },
    ],
    password: [
      {
        required: true,
        trigger: "change",
        message: t("login.message.password.required"),
      },
      {
        min: 6,
        message: t("login.message.password.min"),
        trigger: "blur",
      },
    ],
    captchaCode: [
      {
        required: true,
        trigger: "change",
        message: t("login.message.captchaCode.required"),
      },
    ],
  };
});

// 登录提交处理
async function handleLoginSubmit() {
  try {
    // 1. 表单验证
    if (loginFormData.value.username !== "" && loginFormData.value.password !== "") {
      loading.value = true;
      // 2. 执行登录
      await userStore.login(loginFormData.value);

      // 3. 获取用户信息
      // await userStore.getUserInfo();

      // 4. 解析并跳转目标地址
      const redirect = resolveRedirectTarget(route.query);
      await router.push(redirect);
    } else {
      ElMessage.warning("请填写完整数据！");
    }
  } catch (error) {
    // 5. 统一错误处理
    console.error("登录失败:", error);
  } finally {
    loading.value = false;
  }
}

/**
 * 解析重定向目标
 * @param query 路由查询参数
 * @returns 标准化后的路由地址对象
 */
function resolveRedirectTarget(query) {
  // 默认跳转路径
  const defaultPath = "/jczntgsy";

  // 获取原始重定向路径
  const rawRedirect = query.redirect || defaultPath;

  try {
    // 6. 使用Vue Router解析路径
    const resolved = router.resolve(rawRedirect);
    return {
      path: resolved.path,
      query: resolved.query,
    };
  } catch {
    // 7. 异常处理：返回安全路径
    return { path: defaultPath };
  }
}
</script>

<style lang="scss" scoped>
.login {
  display: flex;
  width: 100vw;
  height: 100vh;
  background: url("@/assets/images/sypt_bg.png") center center no-repeat;
  background-size: 100% 100%;
  .login-img {
    display: flex;
    flex-direction: column;
    margin: 364px 0 0 511px;
    img {
      width: 432px;
      height: 345px;
    }
  }
  .login-con {
    display: flex;
    flex-direction: column;
    width: 380px;
    height: 500px;
    margin: 290px auto 0 187px;
    background: #ffffff;
    box-shadow: 0px 8px 16px 0px rgba(0, 0, 0, 0.08);
    p {
      margin: 0;
      padding: 0;
    }
    .login-title {
      height: 48px;
      margin: 48px auto 0 32px;
      font-family:
        PingFangSC,
        PingFang SC;
      font-weight: 600;
      font-size: 34px;
      color: #02538a;
      line-height: 48px;
    }
    .login-name,
    .login-pass {
      display: flex;
      align-items: center;
      height: 40px;
      margin: 45px auto 0 32px;
      img {
        width: 20px;
        height: 20px;
      }
      :deep(.el-input__wrapper) {
        box-shadow: none;
        border-bottom: 1px solid #eaeaea;
      }
      :deep(.el-input__wrapper.is-focus) {
        box-shadow: none;
        border-bottom: 1px solid #eaeaea;
      }
      .el-input {
        width: 286px;
        margin-left: 10px;
        border: none;
      }
    }
    .login-pass {
      margin-top: 26px;
    }
    .login-btn {
      display: flex;
      justify-content: center;
      align-items: center;
      width: 316px;
      height: 48px;
      margin: 64px auto 0 auto;
      background: #02538a;
      font-family:
        PingFangSC,
        PingFang SC;
      font-weight: 600;
      font-size: 18px;
      color: #ffffff;
      cursor: pointer;
    }
  }
}
</style>
