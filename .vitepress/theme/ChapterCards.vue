<template>
  <div class="content-section">
    <div class="chapter-grid">
      <div class="chapter-card" v-for="chapter in chapters" :key="chapter.id">
        <h2 class="chapter-title">
          <span class="chapter-icon">{{ chapter.id }}.</span>
          {{ chapter.title }}
        </h2>
        <ul class="chapter-list">
          <li class="chapter-item" v-for="(item, index) in chapter.items" :key="index">
            <a :href="withBase(item.link)" class="chapter-link">
              {{ item.text }}
            </a>
          </li>
        </ul>
      </div>
    </div>

    <div class="footer-note">
      <p>欢迎 Star ⭐️ 和贡献！访问 <a href="https://github.com/kebijuelun/Awesome-LLM-Learning" target="_blank">GitHub 仓库</a></p>
      <p style="margin-top: 16px; font-size: 14px;">💡 提示：点击上方任意章节可查看详细内容，使用侧边栏浏览所有小节</p>
    </div>
  </div>
</template>

<script setup>
import { withBase } from 'vitepress'

const chapters = [
  {
    id: 1,
    title: '深度学习基础知识',
    items: [
      { text: '1. Transformer基础', link: '/1.深度学习基础知识/1.Transformer基础' },
      { text: '2. 深度神经网络基础', link: '/1.深度学习基础知识/2.深度神经网络基础' }
    ]
  },
  {
    id: 2,
    title: '自然语言处理基础知识',
    items: [
      { text: '1. 分词器 (Tokenizer)', link: '/2.自然语言处理基础知识/1.分词器(Tokenizer)' },
      { text: '2. 经典NLP模型', link: '/2.自然语言处理基础知识/2.经典NLP模型' },
      { text: '3. 困惑度 (Perplexity)', link: '/2.自然语言处理基础知识/3.困惑度(perplexity)' }
    ]
  },
  {
    id: 3,
    title: '大语言模型基础知识',
    items: [
      { text: '1. 训练框架介绍 (Megatron-lm、DeepSpeed)', link: '/3.大语言模型基础知识/1.训练框架介绍(Megatron-lm、DeepSpeed)' },
      { text: '2. 参数高效微调 (PEFT)', link: '/3.大语言模型基础知识/2.参数高效微调(PEFT)' },
      { text: '3. 经典开源LLM介绍', link: '/3.大语言模型基础知识/3.经典开源LLM介绍' },
      { text: '4. RLHF介绍', link: '/3.大语言模型基础知识/4.RLHF介绍' },
      { text: '5. CoT、ToT介绍', link: '/3.大语言模型基础知识/5.CoT、ToT介绍' },
      { text: '6. SFT训练', link: '/3.大语言模型基础知识/6.SFT训练' },
      { text: '7. 混合专家模型 (MOE)', link: '/3.大语言模型基础知识/7.混合专家模型(MOE)' }
    ]
  },
  {
    id: 4,
    title: '大语言模型推理',
    items: [
      { text: '1. Huggingface推理参数介绍', link: '/4.大语言模型推理/1.Huggingface推理参数介绍' },
      { text: '2. KVCache', link: '/4.大语言模型推理/2.KVCache' },
      { text: '3. LLM推理成本介绍', link: '/4.大语言模型推理/3.LLM推理成本介绍' }
    ]
  },
  {
    id: 5,
    title: '大语言模型应用',
    items: [
      { text: '1. LangChain介绍', link: '/5.大语言模型应用/1.LangChain介绍' }
    ]
  },
  {
    id: 6,
    title: '大语言模型前沿分享',
    items: [
      { text: '1. LLM相关博客分享', link: '/6.大语言模型前沿分享/1.LLM相关博客分享' },
      { text: '2. LLM相关论文分享', link: '/6.大语言模型前沿分享/2.LLM相关论文分享' }
    ]
  }
]
</script>

<style scoped>
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

