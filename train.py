import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from PIL import Image
from dataset import BlurDataset
from datetime import datetime
import numpy as np
import logging
import argparse

ROOT_DIR = "/home/dm/KelingYaoDM/Blurry_image_classifier/blur_dataset"
RESULT_DIR = "/home/dm/KelingYaoDM/Blurry_image_classifier/results"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
CHECKPOINT_DIR = os.path.join(
    RESULT_DIR, timestamp, "checkpoints"
)
OUTPUT_IMAGE_DIR = os.path.join(
    RESULT_DIR, timestamp, "output_images"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Imageet pretrain model settings
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # ImageNet settings
    ]
)

def setup_logging():
    # Create results directory
    results_dir = os.path.join(RESULT_DIR, timestamp)

    # Setup logging
    logging.basicConfig(filename=os.path.join(results_dir, 'training.log'),
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger()

def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    file_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_{}.pth'.format(epoch))
    torch.save(checkpoint, file_path)

def denormalize(tensor, mean, std):
    """
    Denormalize a tensor given the mean and std.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def save_output_images(images, labels, predictions, output_dir):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(len(images)):
        img = images[i].cpu()
        img = denormalize(img, mean, std)
        img = img.numpy().transpose((1, 2, 0))  # (C, H, W) to (H, W, C)
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        label = labels[i].item()
        prediction = predictions[i].item()
        img.save(os.path.join(output_dir, f"label_{label}_prediction_{prediction}.png"))


def train(model, train_loader, checkpoint_dir,logger, num_epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': model.features.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ])
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        save_checkpoint(model, optimizer, epoch, checkpoint_dir)

        scheduler.step(epoch_loss)


def evaluate(model, test_loader, out_image_dir,logger):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            save_output_images(images, labels, predicted, out_image_dir)

    f1 = f1_score(all_labels, all_predictions)
    logger.info(f'F1 Score: {f1:.4f}')

    correct = sum(p == l for p, l in zip(all_predictions, all_labels))
    total = len(all_labels)
    accuracy = 100 * correct / total
    logger.info(f'Accuracy of the model on the test images: {accuracy:.2f}%')

def main():
    parser = argparse.ArgumentParser(description="Blurry Image Classifier Training")
    parser.add_argument('--epoch', type=int, required=True, help='Num of epoch of training process')
    parser.add_argument('--model_type', type=str, required=True, help='B# of efficientNet')
    args = parser.parse_args()
    logger = setup_logging()
    logger.info("Training started")

    dataset = BlurDataset(root_dir=ROOT_DIR, transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    trian_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)
    
    if args.model_type == '0':
        from model import BCNN
        bcnn = BCNN(device=device)
        logger.info(f"Load pretrain model: {args.model_type} ")
        model = bcnn.get_model()
    elif args.model_type == '1':
        from model_small import BCNN
        bcnn = BCNN(device=device)
        logger.info(f"Load pretrain model: {args.model_type} ")
        model = bcnn.get_model()

    elif args.model_type == '2':
        from model_design import BCNN
        model = BCNN()
        logger.info(f"Load pretrain model: {args.model_type} ")


    train(model, trian_loader, CHECKPOINT_DIR,logger,num_epochs=args.epoch)
    evaluate(model, test_loader, OUTPUT_IMAGE_DIR,logger)
    logger.info("Training completed")


if __name__ == "__main__":
    main()
