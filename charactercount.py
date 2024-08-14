import collections
import os
import re


def count_chinese_characters_in_files(file_paths):
    total_text = ""

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                total_text += text
        except Exception as e:
            print(f'文件 {file_path} 打开错误！请重试！')
            print(e)
            continue

    # 使用正则表达式过滤非中文字符，包括常用汉字和生僻字
    chinese_text = re.findall(
        r'[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf\U0002ceb0-\U0002ebef]',
        total_text)

    # 使用Counter统计字符出现次数
    character_count = collections.Counter(chinese_text)

    # 打印统计结果
    print("中文字符出现次数:")
    for character, count in character_count.items():
        print(f"'{character}': {count} 次")


if __name__ == "__main__":
    directory_path = r"/path/to/your/directory"
    file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.txt')]
    count_chinese_characters_in_files(file_paths)
