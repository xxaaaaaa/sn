import os

import cv2
import torch
from PIL import Image
from cnradical import Radical, RunOption

from siamese import Siamese


def list_max(list):
    # 假设第一个最大，最大值的下标0
    index = 0
    max = list[0]
    for i in range(1, len(list)):
        if (list[i] > max):
            max = list[i]
            index = i
    return index, max


def sort(image, path):
    zong = os.listdir(path)  # 取路径下的文件名，生成列表
    # print(len(zong))
    biao = []
    mz = []
    for image_2 in zong:
        if image_2.endswith('.png'):
            mz.append(image_2)
            image_2 = Image.open(path + '/' + image_2)
            probability = model.detect_image(image, image_2)
            biao.append(probability.detach().cpu())

    p, indexes = torch.tensor(biao).topk(4)
    for i in range(len(indexes)):
        radical_out = radical.trans_str(mz[indexes[i]][0])
        print("第%d个；最大相似度：%.2f%%, 名字%s" % (indexes[i], p[i], mz[indexes[i]]))
        print("部首：" + radical_out)
        cv2.imshow('', path + '/' + mz[indexes[i]])


if __name__ == "__main__":
    model = Siamese()
    radical = Radical(RunOption.Radical)
    # while True:
    # image_1 = input('Input image_1 filename:')
    # image_1 = 'img/Angelic_01.png'
    image_1 = r"C:\Users\86184\Desktop\test\xx.png"
    try:
        image_1 = Image.open(image_1)
    except:
        print('Image_1 Open Error! Try again!')
        # continue

    path1 = r"D:\2023暑假\jgwfuzhongwen"
    path2 = r'D:\2023暑假\xztfzw'
    try:
        sort(image_1, path1)
        # sort(path1)

    except:
        print('Image_2 Open Error! Try again!')
        # continue

# if __name__ == "__main__":
#     model = Siamese()
#
#
#
#     image_1 = 'img/Angelic_01.png'
#
#     try:
#         image_1 = Image.open(image_1)
#     except:
#         print('Image_1 Open Error! Try again!')
#         # continue
#
#
#     image_2 = 'img/Angelic_02.png'
#
#
#     try:
#
#         image_2 = Image.open(image_2)
#         probability = model.detect_image(image_1, image_2)
#
#         print(probability)
#     except:
#         print('Image_2 Open Error! Try again!')
#         # continue
