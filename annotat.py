import os
import json
import cv2
import numpy as np
from sklearn.metrics import f1_score

# 文件夹路径
INFERENCE_PATH = '/home/dm/KelingYaoDM/Blurry_image_classifier/inference/bottlechair'

image_files = [f for f in os.listdir(INFERENCE_PATH) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 用于存储结果的字典
results = {}

# 遍历每一张图片
for image_file in image_files:
    # 读取图片
    image_path = os.path.join(INFERENCE_PATH, image_file)
    image = cv2.imread(image_path)
    
    # 显示图片
    cv2.imshow('Image', image)
    key = cv2.waitKey(0)  # 等待键盘输入
    file_name = os.path.splitext(image_file)[0]
    if key == ord('0'):
        results[file_name] = 0
    elif key == ord('1'):
        results[file_name] = 1
    
    # 按Enter键继续下一张图片
    if key == 13:  # Enter key
        continue
    elif key == 27:  # Esc key to exit
        break

# 关闭所有OpenCV窗口
cv2.destroyAllWindows()

# 保存结果到JSON文件
with open(os.path.join(INFERENCE_PATH, 'results.json'), 'w') as json_file:
    json.dump(results, json_file)