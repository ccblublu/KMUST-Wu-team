import os
import numpy as np
import cv2

def add_gaussian_noise(image, mean=0, std=1):
    h, w, c = image.shape
    noise = np.random.normal(mean, std, (h, w, c)).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

# 指定原始图像文件夹和保存噪声图像的文件夹
input_folder = 'images'
output_folder = 'gaosi'

# 确保保存噪声图像的文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历原始图像文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # 读取原始图像
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # 添加高斯噪声
        noisy_image = add_gaussian_noise(image, mean=0, std=10)

        # 保存噪声图像
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, noisy_image)

        print(f"Noise added to {filename}")

print("Noise addition completed.")
