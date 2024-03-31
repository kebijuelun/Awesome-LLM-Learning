import os
import re

def replace_image_back(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content_str = f.read()
        # 定义用于从HTML格式转换回Markdown格式的正则表达式
        pattern = r'<p align="center">\s*<img width="900" alt="(.*?)" src="(.*?)">\s*</p>'
        replacement = r"![\1](\2)"

        # 执行替换
        result = re.sub(pattern, replacement, content_str).strip("\t").strip()
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(result)

def read_markdown_files_back(directory):
    for root, _, files in sorted(os.walk(directory)):
        for file in sorted(files):
            if file.endswith(".md"):
                filepath = os.path.join(root, file)
                replace_image_back(filepath)

if __name__ == "__main__":
    # 指定要遍历的目录
    directory_to_search = "./"

    read_markdown_files_back(directory_to_search)

    print("图片路径替换回Markdown格式完成")
