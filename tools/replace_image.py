import os
import re


def replace_image(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content_str = f.read()

    # 使用正则表达式找到所有匹配项
    pattern = r"!\[(.*?)\]\((.*?)\)"

    # 初始化一个空字符串来构建最终的结果
    result = ""
    last_end = 0  # 上一次匹配结束的位置

    for match in re.finditer(pattern, content_str):
        start, end = match.span()  # 当前匹配项的开始和结束位置

        # 检查匹配项之前是否有 \n
        if content_str[last_end:start].endswith("\n\n"):
            # 如果有，则直接添加匹配项
            replacement = match.group()
        else:
            # 如果没有，则在匹配项前添加 \n
            replacement = "\n" + match.group()

        # 将上一次匹配结束位置到当前匹配开始位置之间的文本和当前的替换内容添加到结果字符串中
        result += content_str[last_end:start] + replacement
        last_end = end  # 更新上一次匹配结束的位置为当前匹配结束的位置

    # 添加最后一个匹配项后面的所有文本
    result += content_str[last_end:]

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
