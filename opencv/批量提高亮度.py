import cv2
import os
import numpy as np

# 输入目录和输出目录
input_dir = "/Users/eric/Downloads/mm"
output_dir = "/Users/eric/Downloads/mm/new/"

# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 循环处理输入目录下的所有图片文件
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 假设处理 jpg 和 png 格式的图片
        # 读取图片
        img = cv2.imread(os.path.join(input_dir, filename))

        alpha = 1.5
        beta = 65

        result = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)


        # 保存处理后的图片到输出目录        
        cv2.imwrite(os.path.join(output_dir, filename), result)

        print(f"Processed: {filename}")

print("All images processed.")
