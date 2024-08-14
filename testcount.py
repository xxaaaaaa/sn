import datetime
from collections import defaultdict

from matplotlib import pyplot as plt

from predict_new import *  # 根据你的 sort 函数的实际位置调整导入


# 统计整合四库结果
def main(image_path, directory_paths):
    # 初始化模型和部首处理器
    model = Siamese()
    radical = Radical(RunOption.Radical)
    img1_name = image_path.split('/')[-1]

    try:
        # 打开图像
        image_1 = Image.open(image_path)
    except Exception as e:
        print('Image_1 打开错误！请重试！')
        print(e)
        return

    radical_count = defaultdict(int)
    filename_count = defaultdict(int)

    for directory_path in directory_paths:
        try:
            # 调用 sort 函数
            results = sortresults(image_1, directory_path)

            for i, resultdan in enumerate(results):
                print(f"Top {i + 1} similar image:")
                print("Similarity score: %.2f%%" % (resultdan["similarity_score"]))
                print("Filename:", resultdan["filename"])
                print("部首：" + resultdan["radical"])
                print("字库：" + resultdan["font"])
            # 统计部首和文件名出现次数
            for result in results:
                radical_count[result["radical"]] += 1
                filename_count[result["filename"]] += 1

        except Exception as e:
            print(f'处理目录 {directory_path} 时出错。')
            print(e)

    # 创建记录的内容
    output_text = "部首出现次数:\n"
    for radical, count in radical_count.items():
        output_text += f"{radical}: {count} 次\n"
        print(f"{radical}: {count} 次\n")

    output_text += "\n文件名出现次数:\n"
    for filename, count in filename_count.items():
        output_text += f"{filename}: {count} 次\n"
        print(f"{filename}: {count} 次\n")

    # 创建记录文件夹，如果不存在
    if not os.path.exists('record'):
        os.makedirs('record')

    # 使用当前时间创建文件名
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"record/{current_time}_{img1_name}.txt"

    # 将结果保存到文件
    with open(filename, "w", encoding="utf-8") as file:
        file.write(output_text)

    # 从保存的文件中读取内容
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # 分析读取的数据并绘图
    radical_count_from_file = {}
    filename_count_from_file = {}

    mode = None
    for line in lines:
        if "部首出现次数" in line:
            mode = "radical"
        elif "文件名出现次数" in line:
            mode = "filename"
        elif ": " in line:
            key, value = line.strip().split(": ")
            value = int(value.split(" ")[0])
            if mode == "radical":
                radical_count_from_file[key] = value
            elif mode == "filename":
                filename_count_from_file[key] = value

    # 绘制读取的数据的饼图
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.pie(radical_count_from_file.values(), labels=radical_count_from_file.keys(), autopct='%1.1f%%', startangle=140)
    plt.title('部首出现次数分布')

    plt.subplot(1, 2, 2)
    plt.pie(filename_count_from_file.values(), labels=filename_count_from_file.keys(), autopct='%1.1f%%',
            startangle=140)
    plt.title('文件名出现次数分布')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_1_path = r"/home/data/zcy/source/1origin.png"
    directory_paths = [
        r"/home/data/zcy/source/jgwfuzhongwen",
        r"/home/data/zcy/source/xztfzw",
        r"/home/data/zcy/source/li_trafzw",
        r"/home/data/zcy/source/kai_trafzw"
    ]
    directory_paths_guge = [
        r"/home/data/zcy/source/gugehua/jgw",
        r"/home/data/zcy/source/gugehua/xz",
        r"/home/data/zcy/source/gugehua/li",
        r"/home/data/zcy/source/gugehua/kai"
    ]

    main(image_1_path, directory_paths_guge)
