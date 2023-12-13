import os


def read_markdown_files(directory):
    markdown_files = []

    for root, _, files in sorted(os.walk(directory)):
        for file in sorted(files):
            if file.endswith(".md"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    markdown_files.append((filepath, f.read()))
    return markdown_files


def merge_markdown_files(markdown_files, output_file, content_str):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(
            "# Awesome-LLM-Learning [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)\n"
        )
        f.write("## 引言\n{}\n".format(markdown_files[0][1][0][1]))
        f.write("## 目录\n{}".format(content_str))

        for dir_path, content in markdown_files[1:]:
            chapter_name = os.path.basename(dir_path).split(".")[1]
            f.write(f"## {chapter_name}\n")
            for filepath, sub_content in content:
                sub_chapter_name = (
                    os.path.basename(filepath).split(".md")[0].split(".")[1]
                )
                if sub_chapter_name != "ignore":
                    f.write(f"### {sub_chapter_name}\n")
                if "./images" in sub_content:
                    sub_content = sub_content.replace(
                        "./images",
                        os.path.join(
                            "https://github.com/kebijuelun/Awesome-LLM-Learning/blob/main/",
                            dir_path,
                            "images",
                        ),
                    )
                    sub_content = sub_content.strip()
                f.write(sub_content)
                f.write("\n")  # 添加一个空行，以防止合并后的内容粘在一起


if __name__ == "__main__":
    # 指定要遍历的目录
    directory_to_search = "./"

    # 指定合并后的输出文件
    output_file = "./README.md"

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
        markdown_files_content = read_markdown_files(dir_path)
        all_contents.append((dir_path, markdown_files_content))

    # 目录生成
    content_str = ""
    for content in all_contents[1:]:
        content_str += "- [{}]({})\n".format(content[0], content[0]).replace("./", "")
        for sub_content in content[1]:
            content_str += "\t- [{}]({})\n".format(
                os.path.basename(sub_content[0].replace(".md", "")), sub_content[0]
            ).replace("./", "")

    # 合并所有Markdown文件内容到一个文件
    merge_markdown_files(all_contents, output_file, content_str)

    print("Markdown文件合并完成！")
