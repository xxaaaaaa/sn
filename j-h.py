import os
import time

from PIL import Image

from siamese import Siamese


def calculate_similarity(image1, image2, model):
    """
    Calculate the similarity between two images using the provided Siamese model.
    """
    probability = model.detect_image(image1, image2)
    return probability.detach().cpu().item()


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


def process_images(jgwfzw_path, sample_path, output_file):
    """
    Process images to find the most similar image from sample folders for each image in jgwfzw folder.
    """
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

        for folder_name in os.listdir(sample_path):
            folder_path = os.path.join(sample_path, folder_name)
            if os.path.isdir(folder_path):
                image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
                selected_images = image_files[:2]  # Select the first two images in the folder

                for img2_name in selected_images:
                    img2_path = os.path.join(folder_path, img2_name)
                    img2 = Image.open(img2_path)
                    similarity = calculate_similarity(img1, img2, model)

                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match = img2_name

        if best_match:
            result = f"{img1_name}, {best_match}, {max_similarity:.2f}"
            append_result_to_file(result, output_file)
            print(f"Processed {img1_name}: {result}")
            end_time = time.time()
            total_time = end_time - start_time
            print(f"用时 {total_time:.4f} 秒")


if __name__ == "__main__":
    jgwfzw_path = "/home/data/zcy/source/gugehua/jgw"
    sample_path = "/home/data/zcy/source/HWOBC数据集/sample"
    output_file = "j-h.txt"

    process_images(jgwfzw_path, sample_path, output_file)
