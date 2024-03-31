import re

# 要匹配的字符串
html_str = '<img width="900" alt="数据并行与模型并行示意图" src="megatron1.png">'

# 正则表达式
pattern = r'src="(.*?)"'

# 匹配结果
match = re.findall(pattern, html_str)

# 输出匹配结果
if match:
    print(match[0])
else:
    print("未匹配到")
