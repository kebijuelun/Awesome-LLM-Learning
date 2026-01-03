---
layout: home

hero:
  name: "Awesome LLM Learning"
  text: "大语言模型学习资料汇总"
  tagline: 专注于大语言模型学习的完整知识体系
  image:
    src: /Awesome-LLM-Learning/.vitepress/theme/logo.svg
    alt: Awesome LLM Learning
  actions:
    - theme: brand
      text: 开始学习
      link: /Awesome-LLM-Learning/1.深度学习基础知识/1.Transformer基础
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

.content-section {
  max-width: 1200px;
  margin: 0 auto;
  padding: 48px 24px;
}

.chapter-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
  gap: 32px;
  margin-top: 32px;
}

@media (max-width: 768px) {
  .chapter-grid {
    grid-template-columns: 1fr;
  }
}

.chapter-card {
  background: var(--vp-c-bg-soft);
  border-radius: 16px;
  padding: 32px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border: 1px solid var(--vp-c-divider);
}

.chapter-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1);
  border-color: var(--vp-c-brand-1);
}

.chapter-title {
  font-size: 28px;
  font-weight: 700;
  color: var(--vp-c-brand-1);
  margin-bottom: 24px;
  display: flex;
  align-items: center;
  gap: 12px;
}

.chapter-icon {
  font-size: 32px;
}

.chapter-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.chapter-item {
  margin-bottom: 16px;
}

.chapter-link {
  display: block;
  font-size: 18px;
  font-weight: 500;
  color: var(--vp-c-text-2);
  text-decoration: none;
  padding: 12px 16px;
  border-radius: 8px;
  transition: all 0.2s;
  background: var(--vp-c-bg);
}

.chapter-link:hover {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  transform: translateX(4px);
}

.footer-note {
  text-align: center;
  margin-top: 64px;
  padding: 32px;
  font-size: 16px;
  color: var(--vp-c-text-2);
}

.footer-note a {
  color: var(--vp-c-brand-1);
  text-decoration: none;
  font-weight: 600;
}

.footer-note a:hover {
  text-decoration: underline;
}
</style>

<div class="content-section">

<div class="chapter-grid">

<div class="chapter-card">
  <h2 class="chapter-title">
    <span class="chapter-icon">1.</span>
    深度学习基础知识
  </h2>
  <ul class="chapter-list">
    <li class="chapter-item">
      <a href="/Awesome-LLM-Learning/1.深度学习基础知识/1.Transformer基础" class="chapter-link">1. Transformer基础</a>
    </li>
    <li class="chapter-item">
      <a href="/Awesome-LLM-Learning/1.深度学习基础知识/2.深度神经网络基础" class="chapter-link">2. 深度神经网络基础</a>
    </li>
  </ul>
</div>

<div class="chapter-card">
  <h2 class="chapter-title">
    <span class="chapter-icon">2.</span>
    自然语言处理基础知识
  </h2>
  <ul class="chapter-list">
    <li class="chapter-item">
      <a href="/Awesome-LLM-Learning/2.自然语言处理基础知识/1.分词器(Tokenizer)" class="chapter-link">1. 分词器 (Tokenizer)</a>
    </li>
    <li class="chapter-item">
      <a href="/Awesome-LLM-Learning/2.自然语言处理基础知识/2.经典NLP模型" class="chapter-link">2. 经典NLP模型</a>
    </li>
    <li class="chapter-item">
      <a href="/Awesome-LLM-Learning/2.自然语言处理基础知识/3.困惑度(perplexity)" class="chapter-link">3. 困惑度 (Perplexity)</a>
    </li>
  </ul>
</div>

<div class="chapter-card">
  <h2 class="chapter-title">
    <span class="chapter-icon">3.</span>
    大语言模型基础知识
  </h2>
  <ul class="chapter-list">
    <li class="chapter-item">
      <a href="/Awesome-LLM-Learning/3.大语言模型基础知识/1.训练框架介绍(Megatron-lm、DeepSpeed)" class="chapter-link">1. 训练框架介绍 (Megatron-lm、DeepSpeed)</a>
    </li>
    <li class="chapter-item">
      <a href="/Awesome-LLM-Learning/3.大语言模型基础知识/2.参数高效微调(PEFT)" class="chapter-link">2. 参数高效微调 (PEFT)</a>
    </li>
    <li class="chapter-item">
      <a href="/Awesome-LLM-Learning/3.大语言模型基础知识/3.经典开源LLM介绍" class="chapter-link">3. 经典开源LLM介绍</a>
    </li>
    <li class="chapter-item">
      <a href="/Awesome-LLM-Learning/3.大语言模型基础知识/4.RLHF介绍" class="chapter-link">4. RLHF介绍</a>
    </li>
    <li class="chapter-item">
      <a href="/Awesome-LLM-Learning/3.大语言模型基础知识/5.CoT、ToT介绍" class="chapter-link">5. CoT、ToT介绍</a>
    </li>
    <li class="chapter-item">
      <a href="/Awesome-LLM-Learning/3.大语言模型基础知识/6.SFT训练" class="chapter-link">6. SFT训练</a>
    </li>
    <li class="chapter-item">
      <a href="/Awesome-LLM-Learning/3.大语言模型基础知识/7.混合专家模型(MOE)" class="chapter-link">7. 混合专家模型 (MOE)</a>
    </li>
  </ul>
</div>

<div class="chapter-card">
  <h2 class="chapter-title">
    <span class="chapter-icon">4.</span>
    大语言模型推理
  </h2>
  <ul class="chapter-list">
    <li class="chapter-item">
      <a href="/Awesome-LLM-Learning/4.大语言模型推理/1.Huggingface推理参数介绍" class="chapter-link">1. Huggingface推理参数介绍</a>
    </li>
    <li class="chapter-item">
      <a href="/Awesome-LLM-Learning/4.大语言模型推理/2.KVCache" class="chapter-link">2. KVCache</a>
    </li>
    <li class="chapter-item">
      <a href="/Awesome-LLM-Learning/4.大语言模型推理/3.LLM推理成本介绍" class="chapter-link">3. LLM推理成本介绍</a>
    </li>
  </ul>
</div>

<div class="chapter-card">
  <h2 class="chapter-title">
    <span class="chapter-icon">5.</span>
    大语言模型应用
  </h2>
  <ul class="chapter-list">
    <li class="chapter-item">
      <a href="/Awesome-LLM-Learning/5.大语言模型应用/1.LangChain介绍" class="chapter-link">1. LangChain介绍</a>
    </li>
  </ul>
</div>

<div class="chapter-card">
  <h2 class="chapter-title">
    <span class="chapter-icon">6.</span>
    大语言模型前沿分享
  </h2>
  <ul class="chapter-list">
    <li class="chapter-item">
      <a href="/Awesome-LLM-Learning/6.大语言模型前沿分享/1.LLM相关博客分享" class="chapter-link">1. LLM相关博客分享</a>
    </li>
    <li class="chapter-item">
      <a href="/Awesome-LLM-Learning/6.大语言模型前沿分享/2.LLM相关论文分享" class="chapter-link">2. LLM相关论文分享</a>
    </li>
  </ul>
</div>

</div>

<div class="footer-note">
  <p>欢迎 Star ⭐️ 和贡献！访问 <a href="https://github.com/kebijuelun/Awesome-LLM-Learning" target="_blank">GitHub 仓库</a></p>
  <p style="margin-top: 16px; font-size: 14px;">💡 提示：点击上方任意章节可查看详细内容，使用侧边栏浏览所有小节</p>
</div>

</div>
