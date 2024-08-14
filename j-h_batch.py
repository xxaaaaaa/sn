import os
import time

from PIL import Image

from siamese_batch import Siamese


def calculate_similarity(image1, image2, model):
    """
    Calculate the similarity between two images using the provided Siamese model.
    """
    probability = model.detect_image(image1, image2)
    return probability.detach().cpu().item()


def calculate_similarity_batch(images1, image2, model, batch_size):
    """
    Calculate the similarity between two images using the provided Siamese model.
    """

    probabilities = model.detect_image_batch(images1, image2, batch_size)

    # 遍历 probabilities 列表，逐个进行 detach 操作并转换为 float
    # probabilities = [prob.detach().cpu().item() for prob in probabilities]

    return probabilities


def load_processed_images(output_file):
    """
    Load the already processed images from the output file.
    """
    if not os.path.exists(output_file):
        return set()

    processed_images = set()
    with open(output_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            img1_name = line.split(",")[0].strip()
            processed_images.add(img1_name)
    return processed_images


def append_result_to_file(result, output_file):
    """
    Append the result to the output file.
    """
    with open(output_file, 'a') as f:
        f.write(result + "\n")


def split_list(lst, batch_size):
    """
    将列表分割为批次
    """
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


def process_images(jgwfzw_path, sample_path, output_file, batch_size):
    # img1=jgw img2=sample
    model = Siamese()

    processed_images = load_processed_images(output_file)

    for img1_name in os.listdir(jgwfzw_path):
        if img1_name in processed_images:
            print(f"Skipping already processed image: {img1_name}")
            continue
        start_time = time.time()
        img1_path = os.path.join(jgwfzw_path, img1_name)
        img1 = Image.open(img1_path)

        max_similarity = -1
        best_match = None

        img2_batches = []

        # 所有
        for folder_name in os.listdir(sample_path):
            folder_path = os.path.join(sample_path, folder_name)
            if os.path.isdir(folder_path):
                image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
                selected_images = image_files[:2]  # 选择前2张图像

                img2_batches.append([os.path.join(folder_path, img2_name) for img2_name in selected_images])
        # 所有img2的地址 7762
        img2_batches = [item for sublist in img2_batches for item in sublist]
        # 7762/32 242
        long = len(img2_batches) // batch_size

        for img2_batch_part in split_list(img2_batches, batch_size):
            a = len(img2_batch_part)
            img2_batch = [Image.open(img2_path) for img2_path in img2_batch_part]
            # img2_batch应该是32/剩余个PIL
            similarities = calculate_similarity_batch(img2_batch, img1, model, a)

            for sim, img2 in zip(similarities, img2_batch):
                img2_name = os.path.basename(img2.filename)  # Get the filename of the image
                if sim > max_similarity:
                    max_similarity = sim
                    best_match = img2_name

            # 关闭所有处理过的图片以释放内存
            for img2 in img2_batch:
                img2.close()

        if best_match:
            result = f"{img1_name}, {best_match}, {max_similarity:.2f}"
            append_result_to_file(result, output_file)
            print(f"Processed {img1_name}: {result}")
            end_time = time.time()
            total_time = end_time - start_time
            print(f"用时 {total_time:.4f} 秒")

        img1.close()


def process_images_classic(img, ku_path, output_file, batch_size):
    # img1=img img2=ku
    model = Siamese()

    processed_images = load_processed_images(output_file)

    for img1_name in os.listdir(ku_path):
        if img1_name in processed_images:
            print(f"Skipping already processed image: {img1_name}")
            continue
        start_time = time.time()
        img1_path = os.path.join(jgwfzw_path, img1_name)
        img1 = Image.open(img1_path)

        max_similarity = -1
        best_match = None

        img2_batches = []

        # 所有
        for folder_name in os.listdir(sample_path):
            folder_path = os.path.join(sample_path, folder_name)
            if os.path.isdir(folder_path):
                image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
                selected_images = image_files[:2]  # 选择前2张图像

                img2_batches.append([os.path.join(folder_path, img2_name) for img2_name in selected_images])
        # 所有img2的地址 7762
        img2_batches = [item for sublist in img2_batches for item in sublist]
        # 7762/32 242
        long = len(img2_batches) // batch_size

        for img2_batch_part in split_list(img2_batches, batch_size):
            a = len(img2_batch_part)
            img2_batch = [Image.open(img2_path) for img2_path in img2_batch_part]
            # img2_batch应该是32/剩余个PIL
            similarities = calculate_similarity_batch(img2_batch, img1, model, a)

            for sim, img2 in zip(similarities, img2_batch):
                img2_name = os.path.basename(img2.filename)  # Get the filename of the image
                if sim > max_similarity:
                    max_similarity = sim
                    best_match = img2_name

            # 关闭所有处理过的图片以释放内存
            for img2 in img2_batch:
                img2.close()

        if best_match:
            result = f"{img1_name}, {best_match}, {max_similarity:.2f}"
            append_result_to_file(result, output_file)
            print(f"Processed {img1_name}: {result}")
            end_time = time.time()
            total_time = end_time - start_time
            print(f"用时 {total_time:.4f} 秒")

        img1.close()


if __name__ == "__main__":
    jgwfzw_path = "/home/data/zcy/source/gugehua/jgw"
    sample_path = "/home/data/zcy/source/HWOBC数据集/sample"
    output_file = "j-h.txt"

    process_images(jgwfzw_path, sample_path, output_file, 512)
