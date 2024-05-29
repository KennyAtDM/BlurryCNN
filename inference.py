# python inference.py --checkpoint /path/to/checkpoint.pth --image /path/to/image.jpg --device cuda

import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
import time
from sklearn.metrics import f1_score, precision_score, recall_score
import json

INFERENCE_PATH = '/home/dm/KelingYaoDM/Blurry_image_classifier/inference/bottlechair'

def load_model(checkpoint_path, device='cuda',model_type = '0'):
    if model_type == '0':
        from model import BCNN
        bcnn = BCNN(device=device)
        model = bcnn.get_model()
    elif model_type == '1':
        from model_small import BCNN
        bcnn = BCNN(device=device)
        model = bcnn.get_model()

    elif model_type == '2':
        from model_design import BCNN
        model = BCNN()
    
    # Load model weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, bcnn.get_transforms()

def predict(image_path, model, transform, device='cuda'):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)

    return predicted.item()
    
    # return 'blurry' if predicted.item() == 1 else 'sharp'

def measure_inference_speed(image_path, model, transform, device='cuda', num_runs=100):
    start_time = time.time()
    for _ in range(num_runs):
        predict(image_path, model, transform, device)
    end_time = time.time()
    total_time = end_time - start_time
    fps = num_runs / total_time
    return fps

def calculate_f1_loss(results, true_labels):
    y_true = [true_labels[image] for image in results.keys()]
    y_pred = list(results.values())
    f1 = f1_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    return f1, precision, recall

def main():
    parser = argparse.ArgumentParser(description="Blurry Image Classifier Inference")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint .pth file')
    parser.add_argument('--model_type', type=str, required=True)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # ImageNet settings
    ])

    # Load model and transform
    model, _ = load_model(args.checkpoint, device, args.model_type)


    # Get the list of images
    image_files = [f for f in os.listdir(INFERENCE_PATH) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Run predictions for each image
    results = {}
    for image_file in image_files:
        result = predict(os.path.join(INFERENCE_PATH, image_file), model, transform, device)
        results[os.path.splitext(image_file)[0]] = result

    true_labels = {} 
    with open(os.path.join(INFERENCE_PATH, 'results.json'), 'r') as json_file:
        true_labels = json.load(json_file)

    f1, precision, recall = calculate_f1_loss(results, true_labels)
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    # fps = measure_inference_speed(image_file, model, transform, device)
    # print(f"Inference speed: {fps:.2f} FPS")

if __name__ == '__main__':
    main()
