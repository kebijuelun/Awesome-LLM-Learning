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

features:
  - icon: 1️⃣
    title: 深度学习基础知识
    details: |
      [1. Transformer基础](/1.深度学习基础知识/1.Transformer基础)  
      [2. 深度神经网络基础](/1.深度学习基础知识/2.深度神经网络基础)
    
  - icon: 2️⃣
    title: 自然语言处理基础知识
    details: |
      [1. 分词器 (Tokenizer)](/2.自然语言处理基础知识/1.分词器\(Tokenizer\))  
      [2. 经典NLP模型](/2.自然语言处理基础知识/2.经典NLP模型)  
      [3. 困惑度 (Perplexity)](/2.自然语言处理基础知识/3.困惑度\(perplexity\))

  - icon: 3️⃣
    title: 大语言模型基础知识
    details: |
      [1. 训练框架介绍](/3.大语言模型基础知识/1.训练框架介绍\(Megatron-lm、DeepSpeed\))  
      [2. 参数高效微调 (PEFT)](/3.大语言模型基础知识/2.参数高效微调\(PEFT\))  
      [3. 经典开源LLM介绍](/3.大语言模型基础知识/3.经典开源LLM介绍)  
      [4. RLHF介绍](/3.大语言模型基础知识/4.RLHF介绍)  
      [5. CoT、ToT介绍](/3.大语言模型基础知识/5.CoT、ToT介绍)  
      [6. SFT训练](/3.大语言模型基础知识/6.SFT训练)  
      [7. 混合专家模型 (MOE)](/3.大语言模型基础知识/7.混合专家模型\(MOE\))

  - icon: 4️⃣
    title: 大语言模型推理
    details: |
      [1. Huggingface推理参数介绍](/4.大语言模型推理/1.Huggingface推理参数介绍)  
      [2. KVCache](/4.大语言模型推理/2.KVCache)  
      [3. LLM推理成本介绍](/4.大语言模型推理/3.LLM推理成本介绍)

  - icon: 5️⃣
    title: 大语言模型应用
    details: |
      [1. LangChain介绍](/5.大语言模型应用/1.LangChain介绍)

  - icon: 6️⃣
    title: 大语言模型前沿分享
    details: |
      [1. LLM相关博客分享](/6.大语言模型前沿分享/1.LLM相关博客分享)  
      [2. LLM相关论文分享](/6.大语言模型前沿分享/2.LLM相关论文分享)
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

/* 自定义 features 样式 */
.VPFeatures {
  padding: 48px 24px !important;
}

.VPFeature {
  border-radius: 16px !important;
  padding: 32px !important;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.VPFeature:hover {
  transform: translateY(-4px) !important;
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1) !important;
}

.VPFeature .icon {
  font-size: 48px !important;
}

.VPFeature .title {
  font-size: 24px !important;
  font-weight: 700 !important;
  color: var(--vp-c-brand-1) !important;
  margin: 16px 0 !important;
}

.VPFeature .details a {
  display: block;
  padding: 8px 0;
  color: var(--vp-c-text-2);
  text-decoration: none;
  transition: all 0.2s;
}

.VPFeature .details a:hover {
  color: var(--vp-c-brand-1);
  transform: translateX(4px);
}
</style>
