import os
import shutil
from collections import defaultdict

library1 = r"/home/data/zcy/source/gugehua/jgw"
library2 = r"/home/data/zcy/source/gugehua/xz"
library3 = r"/home/data/zcy/source/gugehua/kai"
library4 = r"/home/data/zcy/source/gugehua/li"

# 四个文字库路径
libraries = [library1, library2, library3, library4]

# 目标文件夹
target_folder = "test"

# 创建目标文件夹
if not os.path.exists(target_folder):
    os.mkdir(target_folder)

# 字符统计字典，key为字符，value为包含此字符的库数量
char_count = defaultdict(int)

# 用于存储字符及其对应的文件路径
char_to_files = defaultdict(list)

# 遍历每个文字库，统计每个库中有哪些字
for library in libraries:
    library_name = os.path.basename(library)  # 提取字库的文件夹名称
    for filename in os.listdir(library):
        if filename.endswith(".png"):
            # 提取文件名中的文字部分
            text = filename[0]  # 文件名的第一个字符为文字
            char_count[text] += 1
            char_to_files[text].append((os.path.join(library, filename), library_name))

# 找出在所有库中都存在的字
common_chars = [char for char, count in char_count.items() if count == len(libraries)]

# 创建文件夹并复制图片
for char in common_chars:
    text_folder = os.path.join(target_folder, char)

    if not os.path.exists(text_folder):
        os.mkdir(text_folder)

    for file_path, library_name in char_to_files[char]:
        # 提取字库名称前面的部分（即去掉_fzw后的部分）
        library_prefix = library_name

        # 生成一个唯一的文件名，加入库名前缀
        new_filename = f"{char}_{library_prefix}.png"
        destination_file = os.path.join(text_folder, new_filename)

        # 复制文件到目标文件夹
        shutil.copy(file_path, destination_file)

print("数据集已生成！")
