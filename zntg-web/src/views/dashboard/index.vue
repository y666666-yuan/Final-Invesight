<template>
  <div class="app-container">
    <div class="zt-top">
      <p>智投Invesight</p>
      <p>—— 让智能投顾创新，从“黑箱探索”走向“导航前行”</p>
    </div>
    <div class="zt-top zt-title">
      <p>智能投顾沙盒实验室</p>
      <p>—— 导航式开发测试平台，赋能金融机构智能投顾创新</p>
    </div>
    <div class="zt-con">
      <p>智投Invesight 是面向金融机构打造的智能投顾沙盒实验室，采用“三位一体”架构，集成约束优化引擎、冷启动适配模块与可解释决策系统三大核心能力，专为提升机构智能投顾研发效率而设计。</p>
      <p>平台支持机构用户上传自有数据，快速导入客户画像与产品库，在贴近真实业务的仿真环境中，一站式完成从数据预处理、特征工程、算法配置到推荐效果评估的全流程测试。通过直观的导航式操作界面，轻松体验多种智能推荐算法，验证其在多约束配置、新用户冷启动、决策透明化等关键场景下的表现。</p>
      <p>智投Invesight 不仅是技术验证平台，更是智能投顾能力孵化的“数字试验场”。依托模块化架构与开放接口，平台为金融机构提供可定制、可追溯、可集成的开发环境，助力实现从算法探索到业务落地的高效闭环。</p>
    </div>
  </div>
</template>

<script setup>
onMounted(() => {});
</script>

<style lang="scss" scoped>
.app-container{
  display: flex;
  flex-direction: column;
  width: calc(100% - 32px);
  height: 100%;
  margin: 0 16px;
  background:
    url('@/assets/images/home_left.png') no-repeat left bottom,
    url('@/assets/images/home_right.png') no-repeat right bottom;
  background-size: auto; /* 可根据需要设置，如 100px auto */
  background-color: #FFFFFF;
  box-shadow: 0px 2px 4px 0px rgba(0,0,0,0.06);
  border-radius: 2px;
  .zt-top{
    display: flex;
    align-items: center;
    width: 1280px;
    height: 72px;
    margin: 64px auto 0 auto;
    padding: 0 60px;
    background: linear-gradient( 180deg, #FCFEFF 0%, #EFF8FF 100%);
    border: 1px solid #E6F5FF;
    p{
      margin: 0;
      padding: 0;
    }
    p:nth-of-type(1){
      min-width: 211px;
      height: 40px;
      font-family: PingFangSC, PingFang SC;
      font-weight: 600;
      font-size: 32px;
      color: #003D67;
      line-height: 40px;
    }
    p:nth-of-type(2){
      height: 28px;
      margin-left: 440px;
      font-family: PingFangSC, PingFang SC;
      font-weight: 400;
      font-size: 20px;
      color: rgba(0,0,0,0.85);
      line-height: 28px;
    }
  }
  .zt-title{
    margin: 32px auto 0 auto;
    p:nth-of-type(2){
      margin-left: 363px;
    }
  }
  .zt-con{
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    width: 1280px;
    height: 324px;
    margin: 32px auto 0 auto;
    padding: 36px 60px;
    background: linear-gradient( 180deg, #FCFEFF 0%, #EFF8FF 100%);
    border: 1px solid #E6F5FF;
    p{
      margin: 0;
      padding: 0;
      font-family: PingFangSC, PingFang SC;
      font-weight: 400;
      font-size: 18px;
      color: rgba(0,0,0,0.85);
      line-height: 28px;
      text-indent: 36px;
    }
  }
}
</style>
