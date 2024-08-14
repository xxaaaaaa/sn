import os

import torch
from PIL import Image
from cnradical import Radical, RunOption

from siamese import Siamese


# 对于HW数据集，返回print
def sort1(image, path):
    model = Siamese()
    radical = Radical(RunOption.Radical)
    filenames = os.listdir(path)
    similarities = []
    for filename in filenames:
        print(filename)
        file_path = os.path.join(path, filename)
        print(file_path)
        imagenames = os.listdir(file_path)
        print(imagenames)
        for imagename in imagenames:
            if imagename.endswith('.png'):
                image_path = os.path.join(path, imagename)
                image_2 = Image.open(image_path)
                # Calculate similarity score
                probability = model.detect_image(image, image_2)
                similarities.append(probability.detach().cpu())

    # Find top 5 similar images
    top_similarities, top_indexes = torch.tensor(similarities).topk(5)
    for i in range(len(top_indexes)):
        print(f"Top {i + 1} similar image:")
        print("Similarity score: %.2f%%" % (top_similarities[i]))
        print("Filename:", filenames[top_indexes[i]])


# 对于附中文字库，返回print
def sort(image, path):
    model = Siamese()
    radical = Radical(RunOption.Radical)

    filenames = os.listdir(path)
    similarities = []
    for filename in filenames:
        if filename.endswith('.png'):
            image_path = os.path.join(path, filename)
            image_2 = Image.open(image_path)
            # Calculate similarity score
            probability = model.detect_image(image, image_2)
            similarities.append(probability.detach().cpu())

    # Find top 5 similar images
    top_similarities, top_indexes = torch.tensor(similarities).topk(5)
    for i in range(len(top_indexes)):
        print(f"Top {i + 1} similar image:")
        print("Similarity score: %.2f%%" % (top_similarities[i]))
        print("Filename:", filenames[top_indexes[i]])
        radical_out = radical.trans_str(filenames[top_indexes[i]][0])
        print("部首：" + radical_out)


# 对于返回结果列表
def sortresults(image, path):
    model = Siamese()
    radical = Radical(RunOption.Radical)
    font = os.path.basename(path)
    filenames = os.listdir(path)
    similarities = []

    for filename in filenames:
        if filename.endswith('.png'):
            image_path = os.path.join(path, filename)
            image_2 = Image.open(image_path)
            # 计算相似度分数
            probability = model.detect_image(image, image_2)
            similarities.append(probability.detach().cpu())

    # 找到最相似的5张图片
    top_similarities, top_indexes = torch.tensor(similarities).topk(5)
    results = []

    for i in range(len(top_indexes)):
        index = top_indexes[i].item()
        similarity_score = top_similarities[i].item()
        filename = filenames[index]
        radical_out = radical.trans_str(filename[0])

        results.append({
            "similarity_score": similarity_score,
            "filename": filename,
            "radical": radical_out,
            "font": font
        })

    return results


# 计算一组 按发展时期排列顺序的同一文字的互相相似度
def calculate_and_save_similarity(path, output_filename="result.txt"):
    # 初始化模型
    model = Siamese()

    # 定义图片文件名中的后缀和比较顺序
    comparisons = [
        ('_jgw', '_xz'),
        ('_xz', '_li'),
        ('_li', '_kai')
    ]

    output_path = os.path.join(path, output_filename)
    existing_results = set()

    # 读取已存在的结果，避免重复计算
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 4):  # 4行一组
                image1 = lines[i].split(":")[1].strip()
                image2 = lines[i + 1].split(":")[1].strip()
                existing_results.add((image1, image2))

    similarities = []

    # 遍历比较对
    for suffix1, suffix2 in comparisons:
        # 找到指定后缀的图片
        filenames = [f for f in os.listdir(path) if f.endswith('.png')]
        filename1 = next((f for f in filenames if suffix1 in f), None)
        filename2 = next((f for f in filenames if suffix2 in f), None)

        # 打印调试信息，确认文件是否被正确找到
        print(f"Comparing {filename1} and {filename2}...")

        if filename1 and filename2:
            # 检查是否已经计算过
            if (filename1, filename2) in existing_results or (filename2, filename1) in existing_results:
                print(f"Skipping already processed pair: {filename1} and {filename2}")
                continue

            image_path1 = os.path.join(path, filename1)
            image_path2 = os.path.join(path, filename2)
            try:
                image1 = Image.open(image_path1)
                image2 = Image.open(image_path2)

                # 计算相似度分数
                probability = model.detect_image(image1, image2)
                similarity_score = probability.detach().cpu().item()
                similarities.append({
                    "image1": filename1,
                    "image2": filename2,
                    "similarity_score": similarity_score
                })

                # 打印相似度得分，确认模型输出
                print(f"Similarity score between {filename1} and {filename2}: {similarity_score:.2f}%")
            except Exception as e:
                print(f"Error processing {image_path1} or {image_path2}: {e}")

    # 将相似度结果写入文本文件
    with open(output_path, 'a') as f:
        for result in similarities:
            f.write(f"Image 1: {result['image1']}\n")
            f.write(f"Image 2: {result['image2']}\n")
            f.write(f"Similarity score: {result['similarity_score']:.2f}%\n")
            f.write("\n")

    print(f"Similarity results saved to {output_path}")


if __name__ == "__main__":

    image_1_path = r"/home/data/zcy/source/yy.png"
    try:
        image_1 = Image.open(image_1_path)

    except Exception as e:
        print('Image_1 打开错误！请重试！')
        print(e)

    directory_path = r"/home/data/zcy/source/jgwfuzhongwen"
    try:
        # sort(image_1, directory_path)
        results = sortresults(image_1, directory_path)

        # 打印结果
        for i, result in enumerate(results):
            print(f"Top {i + 1} similar image:")
            print("Similarity score: %.2f%%" % (result["similarity_score"]))
            print("Filename:", result["filename"])
            print("部首：" + result["radical"])

    except Exception as e:
        print('处理图像时出错。')
        print(e)
