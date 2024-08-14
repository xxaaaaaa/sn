import os
import sys

import cv2
import numpy as np

t = 127

path1 = r'C:\Users\86184\Desktop\xia1-80\1-80\41-80\41-80'
path2 = r'C:\Users\86184\Desktop\xia1-80\1-80\41-80\41-80\70'


def BorWsmp(img, bwThresh):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    imgShape = img.shape
    h = imgShape[0]
    w = imgShape[1]
    blackNum, whiteNum = 0, 0
    blackNum1, whiteNum1 = 0, 0
    x1 = 20
    for y in range(w):
        if gray[x1][y] <= bwThresh:
            blackNum += 1
        else:
            whiteNum += 1
    x2 = h - 20
    for y in range(w):
        if gray[x2][y] <= bwThresh:
            blackNum1 += 1
        else:
            whiteNum1 += 1
    print(blackNum, whiteNum)
    print(blackNum1, whiteNum1)
    if blackNum > whiteNum or blackNum1 > whiteNum1:
        print("black based image")
        return 1
    else:
        print("white based image")
        return 0


# 根据指定x1 x2横线上黑白像素多少判断底色 需要手动设置调教

def BorW(img, bwThresh):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    imgShape = img.shape
    h = imgShape[0]
    w = imgShape[1]
    # init black and white point number
    blackNum, whiteNum = 0, 0
    k = h / w
    for x in range(w):
        y1 = int(k * x)
        y2 = int((-k) * x + h - 1)
        # prevent overflow
        if 0 <= y1 <= (h - 1) and 0 <= y2 <= (h - 1):
            # first diagonal line
            if gray[y1][x] <= bwThresh:
                blackNum += 1
            else:
                whiteNum += 1
            # second diagonal line
            if gray[y2][x] <= bwThresh:
                blackNum += 1
            else:
                whiteNum += 1
    print(blackNum, whiteNum)
    if blackNum > whiteNum:
        print("black based image")
        return 1
    else:
        print("white based image")
        return 0


# 根据对角线上黑白像素多少判断底色 有些情况不准

def testedge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for white areas near the edges
    mask = np.zeros_like(image)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x == 0 or y == 0 or x + w == image.shape[1] or y + h == image.shape[0]:
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    # Replace white areas near the edges with black
    image[np.where((mask == [255, 255, 255]).all(axis=2))] = [0, 0, 0]

    # 显示结果
    # cv2.imshow('Result', image)
    # cv2.waitKey(0)
    return image


# 边缘检测算法


def get_all_folder(path):
    folders = []
    for name in os.listdir(path):
        folder_path = os.path.join(path, name)
        if os.path.isdir(folder_path):
            folders.append(folder_path)
    return folders


# 提取文件路径集合


def replace_files(path):
    old_foldername = path[-3:]
    if old_foldername.startswith('\\'):
        old_foldername = path[-2:]
        folder_name = old_foldername + '_1'
        new_path = os.path.join(path[:-2], folder_name)
    else:
        folder_name = old_foldername + '_1'
        new_path = os.path.join(path[:-3], folder_name)
    print(new_path)
    os.mkdir(new_path)
    old_imgnames = os.listdir(path)  # 取路径下的文件名，生成列表
    for old_imgname in old_imgnames:  # 遍历列表下的文件名
        print(old_imgname)
        if old_imgname != sys.argv[0]:  # 代码本身文件路径，防止脚本文件放在path路径下时，被一起重命名
            if old_imgname.endswith('.png'):
                img1 = cv2.imread(os.path.join(path, old_imgname), cv2.IMREAD_COLOR)
                bow = BorWsmp(img1, t)
                if bow == 1:
                    # Save the result
                    cv2.imwrite(os.path.join(new_path, old_imgname), testedge(img1))
    if len(os.listdir(new_path)) == 0:
        os.rmdir(new_path)
# 未知功能的重命名


# folders_path = get_all_folder(path1)
# # for folder_path in folders_path:
# #     replace_files(folder_path)
# #
# replace_files(path2)
