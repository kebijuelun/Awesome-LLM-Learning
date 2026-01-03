import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Awesome LLM Learning",
  description: "大语言模型学习资料汇总",
  base: '/Awesome-LLM-Learning/',
  lang: 'zh-CN',
  
  head: [
    ['link', { rel: 'icon', href: '/Awesome-LLM-Learning/favicon.ico' }],
    ['meta', { name: 'theme-color', content: '#3eaf7c' }],
    ['meta', { name: 'apple-mobile-web-app-capable', content: 'yes' }],
    ['meta', { name: 'apple-mobile-web-app-status-bar-style', content: 'black' }]
  ],

  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: '首页', link: '/' },
      { text: '深度学习基础', link: '/1.深度学习基础知识/1.Transformer基础' },
      { text: 'NLP基础', link: '/2.自然语言处理基础知识/1.分词器(Tokenizer)' },
      { text: 'LLM基础', link: '/3.大语言模型基础知识/1.训练框架介绍(Megatron-lm、DeepSpeed)' },
      { text: 'LLM推理', link: '/4.大语言模型推理/1.Huggingface推理参数介绍' },
      { text: 'LLM应用', link: '/5.大语言模型应用/1.LangChain介绍' },
      { text: '前沿分享', link: '/6.大语言模型前沿分享/1.LLM相关博客分享' },
      { 
        text: 'GitHub',
        link: 'https://github.com/kebijuelun/Awesome-LLM-Learning'
      }
    ],

    sidebar: {
      '/1.深度学习基础知识/': [
        {
          text: '深度学习基础知识',
          items: [
            { text: 'Transformer基础', link: '/1.深度学习基础知识/1.Transformer基础' },
            { text: '深度神经网络基础', link: '/1.深度学习基础知识/2.深度神经网络基础' }
          ]
        }
      ],
      '/2.自然语言处理基础知识/': [
        {
          text: '自然语言处理基础知识',
          items: [
            { text: '分词器(Tokenizer)', link: '/2.自然语言处理基础知识/1.分词器(Tokenizer)' },
            { text: '经典NLP模型', link: '/2.自然语言处理基础知识/2.经典NLP模型' },
            { text: '困惑度(perplexity)', link: '/2.自然语言处理基础知识/3.困惑度(perplexity)' }
          ]
        }
      ],
      '/3.大语言模型基础知识/': [
        {
          text: '大语言模型基础知识',
          items: [
            { text: '训练框架介绍(Megatron-lm、DeepSpeed)', link: '/3.大语言模型基础知识/1.训练框架介绍(Megatron-lm、DeepSpeed)' },
            { text: '参数高效微调(PEFT)', link: '/3.大语言模型基础知识/2.参数高效微调(PEFT)' },
            { text: '经典开源LLM介绍', link: '/3.大语言模型基础知识/3.经典开源LLM介绍' },
            { text: 'RLHF介绍', link: '/3.大语言模型基础知识/4.RLHF介绍' },
            { text: 'CoT、ToT介绍', link: '/3.大语言模型基础知识/5.CoT、ToT介绍' },
            { text: 'SFT训练', link: '/3.大语言模型基础知识/6.SFT训练' },
            { text: '混合专家模型(MOE)', link: '/3.大语言模型基础知识/7.混合专家模型(MOE)' }
          ]
        }
      ],
      '/4.大语言模型推理/': [
        {
          text: '大语言模型推理',
          items: [
            { text: 'Huggingface推理参数介绍', link: '/4.大语言模型推理/1.Huggingface推理参数介绍' },
            { text: 'KVCache', link: '/4.大语言模型推理/2.KVCache' },
            { text: 'LLM推理成本介绍', link: '/4.大语言模型推理/3.LLM推理成本介绍' }
          ]
        }
      ],
      '/5.大语言模型应用/': [
        {
          text: '大语言模型应用',
          items: [
            { text: 'LangChain介绍', link: '/5.大语言模型应用/1.LangChain介绍' }
          ]
        }
      ],
      '/6.大语言模型前沿分享/': [
        {
          text: '大语言模型前沿分享',
          items: [
            { text: 'LLM相关博客分享', link: '/6.大语言模型前沿分享/1.LLM相关博客分享' },
            { text: 'LLM相关论文分享', link: '/6.大语言模型前沿分享/2.LLM相关论文分享' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/kebijuelun/Awesome-LLM-Learning' }
    ],

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024 Awesome LLM Learning'
    },

    outline: {
      level: [2, 3],
      label: '目录'
    },

    docFooter: {
      prev: '上一页',
      next: '下一页'
    },

    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'short'
      }
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索文档',
            buttonAriaLabel: '搜索文档'
          },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换'
            }
          }
        }
      }
    }
  },

  markdown: {
    lineNumbers: true,
    math: true
  }
})

