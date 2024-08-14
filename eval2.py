import os
import random
import shutil

import pandas as pd
from PIL import Image
from sklearn.metrics import precision_score, recall_score

from siamese_batch import Siamese

# 定义数据集路径
dataset_path = '/home/data/zcy/source/HWOBC数据集/sample'
test_path = '/home/data/zcy/source/test_set'
train_path = '/home/data/zcy/source/train_set'
results_file = 'results.csv'

# 实例化Siamese类
siamese = Siamese(model_path='model_data/best_epoch_weights.pth', cuda=True)


# 准备数据集
def prepare_dataset(dataset_path, test_path, train_path, num_test=7):
    if os.path.exists(test_path) and os.path.exists(train_path):
        print("数据集已经存在，跳过数据集准备步骤。")
        return

    os.makedirs(test_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)

    for character in os.listdir(dataset_path):
        char_path = os.path.join(dataset_path, character)
        images = os.listdir(char_path)
        random.shuffle(images)

        test_images = images[:num_test]
        train_images = images[num_test:]

        os.makedirs(os.path.join(test_path, character), exist_ok=True)
        os.makedirs(os.path.join(train_path, character), exist_ok=True)

        for img in test_images:
            shutil.copy(os.path.join(char_path, img), os.path.join(test_path, character, img))
        for img in train_images:
            shutil.copy(os.path.join(char_path, img), os.path.join(train_path, character, img))


# 准备数据集
prepare_dataset(dataset_path, test_path, train_path)

# 获取所有测试图片路径
test_image_paths = []
for character in os.listdir(test_path):
    char_path = os.path.join(test_path, character)
    for img in os.listdir(char_path):
        test_image_paths.append((character, os.path.join(char_path, img)))

# 加载或初始化结果文件
if os.path.exists(results_file):
    results_df = pd.read_csv(results_file)
else:
    results_df = pd.DataFrame(
        columns=['Test Character', 'Test Image', 'Train Character', 'Train Image', 'Similarity', 'Ground Truth'])


# 计算相似度并保存结果
def calculate_similarity(siamese, test_image_paths, train_path, results_df, results_file, batch_size=512):
    for character, test_image_path in test_image_paths:
        if results_df[(results_df['Test Character'] == character) & (
                results_df['Test Image'] == os.path.basename(test_image_path))].shape[0] > 0:
            print(f"Skipping {os.path.basename(test_image_path)}, already processed.")
            continue  # 如果已经计算过，跳过

        test_image = Image.open(test_image_path)

        # 准备一个批次的训练图片
        train_images = []
        train_chars = []
        train_image_paths = []

        for char in os.listdir(train_path):
            char_path = os.path.join(train_path, char)
            for img in os.listdir(char_path):
                train_image_path = os.path.join(char_path, img)

                # 检查是否已处理过这个组合
                if results_df[(results_df['Test Image'] == os.path.basename(test_image_path)) &
                              (results_df['Train Image'] == os.path.basename(train_image_path))].shape[0] > 0:
                    print(f"Skipping {os.path.basename(test_image_path)}vs{os.path.basename(train_image_path)}, "
                          f"already processed.")
                    continue

                train_image = Image.open(train_image_path)

                train_images.append(train_image)
                train_chars.append(char)
                train_image_paths.append(train_image_path)

                # 如果已经收集到 batch_size 数量的图片，则进行一次批量推理
                if len(train_images) == batch_size:
                    similarities = siamese.detect_image_batch(train_images, test_image, batch_size)

                    for i, similarity in enumerate(similarities):
                        is_same_character = 1 if train_chars[i] == character else 0
                        new_row = {
                            'Test Character': character,
                            'Test Image': os.path.basename(test_image_path),
                            'Train Character': train_chars[i],
                            'Train Image': os.path.basename(train_image_paths[i]),
                            'Similarity': similarity,
                            'Ground Truth': is_same_character
                        }
                        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                        print(
                            f"Processed: {os.path.basename(test_image_path)} vs {os.path.basename(train_image_paths[i])}")

                        # 每次添加结果后保存到CSV文件
                        results_df.tail(1).to_csv(results_file, index=False, mode='a',
                                                  header=not os.path.exists(results_file))

                    # 清空列表，准备下一批图片
                    train_images.clear()
                    train_chars.clear()
                    train_image_paths.clear()

        # 处理剩余的图片（不足一个批次）
        if train_images:
            similarities = siamese.detect_image_batch(train_images, test_image, len(train_images))
            for i, similarity in enumerate(similarities):
                is_same_character = 1 if train_chars[i] == character else 0
                new_row = {
                    'Test Character': character,
                    'Test Image': os.path.basename(test_image_path),
                    'Train Character': train_chars[i],
                    'Train Image': os.path.basename(train_image_paths[i]),
                    'Similarity': similarity,
                    'Ground Truth': is_same_character
                }
                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                print(f"Processed: {os.path.basename(test_image_path)} vs {os.path.basename(train_image_paths[i])}")

                # 每次添加结果后保存到CSV文件
                results_df.tail(1).to_csv(results_file, index=False, mode='a', header=not os.path.exists(results_file))


# 计算相似度并保存结果
calculate_similarity(siamese, test_image_paths, train_path, results_df, results_file)

# 加载计算好的结果
results_df = pd.read_csv(results_file)

# 计算precision和recall
threshold = 99
results_df['Binary Prediction'] = results_df['Similarity'].apply(lambda x: 1 if x >= threshold else 0)

precision = precision_score(results_df['Ground Truth'], results_df['Binary Prediction'])
recall = recall_score(results_df['Ground Truth'], results_df['Binary Prediction'])

# 打印结果
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

# 展示结果
summary = results_df.groupby(['Test Character', 'Test Image']).agg({
    'Ground Truth': 'sum',
    'Binary Prediction': 'sum'
}).reset_index()

summary['Precision'] = precision
summary['Recall'] = recall

print(summary)
