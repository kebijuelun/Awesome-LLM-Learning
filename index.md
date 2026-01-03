---
layout: home

hero:
  name: "Awesome LLM Learning"
  text: "大语言模型学习资料汇总"
  tagline: 专注于大语言模型学习的完整知识体系
  image:
    src: /.vitepress/theme/logo.svg
    alt: Awesome LLM Learning
  actions:
    - theme: brand
      text: 开始学习
      link: /1.深度学习基础知识/1.Transformer基础
    - theme: alt
      text: GitHub
      link: https://github.com/kebijuelun/Awesome-LLM-Learning
---

<style>
.VPHero .image-bg {
  opacity: 0.4;
}

/* 调整首页标题字体大小 */
.VPHero .name {
  font-size: 48px !important;
  line-height: 1.5 !important;
  padding: 8px 0 !important;
}

.VPHero .text {
  font-size: 32px !important;
  line-height: 1.5 !important;
  padding: 4px 0 !important;
}

@media (max-width: 768px) {
  .VPHero .name {
    font-size: 36px !important;
  }
  
  .VPHero .text {
    font-size: 24px !important;
  }
}
</style>

<ChapterCards />
