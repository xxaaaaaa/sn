import os
import time

from predict_new import calculate_and_save_similarity

test_path = r"/home/data/zcy/project1/test"

for cha_name in os.listdir(test_path):
    cha_path = os.path.join(test_path, cha_name)
    print(cha_path)
    start = time.time()
    calculate_and_save_similarity(cha_path, output_filename="result.txt")
    end = time.time()
    total_time = end - start
    print(os.path.basename(cha_path))
    print(f"cost:{total_time:.2f}s")
