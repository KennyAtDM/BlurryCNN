import cv2
import os

INFERENCE_PATH = '/home/dm/KelingYaoDM/Blurry_image_classifier/inference'


def check_image_is_vague(image, threshold=100):
    """
    检查照片是否模糊
    :param image: 照片地址
    :param threshold: 门限
    :return: True或False
    """
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    print('image vague is {}'.format(fm))
    if fm > threshold:
        return True
    return False


if __name__ == '__main__':
    image_files = [f for f in os.listdir(INFERENCE_PATH) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Run predictions for each image
    results = {}
    for image_file in image_files:
        result = check_image_is_vague(os.path.join(INFERENCE_PATH, image_file))
        results[os.path.splitext(image_file)[0]] = result
    print(results)