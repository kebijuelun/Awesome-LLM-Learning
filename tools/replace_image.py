import os
import re


def replace_image(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content_str = f.read()
        # 使用正则表达式进行替换
        pattern = r"!\[(.*?)\]\((.*?)\)"
        replacement = r'<p align="center"> \n \t <img width="900" alt="\1" src="\2"> \n </p> \n'

        # 执行替换
        result = re.sub(pattern, replacement, content_str).strip("\t").strip()
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(result)


def read_markdown_files(directory):
    for root, _, files in sorted(os.walk(directory)):
        for file in sorted(files):
            if file.endswith(".md"):
                filepath = os.path.join(root, file)
                replace_image(filepath)


if __name__ == "__main__":
    # 指定要遍历的目录
    directory_to_search = "./"

    # 获取目录中的文件夹
    exclude_dir_names = [".git", "tools"]
    dir_names = sorted(os.listdir(directory_to_search))
    dir_paths = [
        os.path.join(directory_to_search, dir_name)
        for dir_name in dir_names
        if os.path.isdir(os.path.join(directory_to_search, dir_name))
        and dir_name not in exclude_dir_names
    ]

    all_contents = []
    # 获取目录中的Markdown文件内容
    for dir_path in dir_paths:
        read_markdown_files(dir_path)

    print("图片路径替换完成")
