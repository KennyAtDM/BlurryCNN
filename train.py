import os
import torch
import random
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from dataset import BlurDataset
from datetime import datetime
import logging
import argparse
from collections import Counter

ROOT_DIR = "/home/dm/KelingYaoDM/Blurry_image_classifier/blur_dataset"
RESULT_DIR = "/home/dm/KelingYaoDM/Blurry_image_classifier/results"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
CHECKPOINT_DIR = os.path.join(
    RESULT_DIR, timestamp, "checkpoints"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def adjust_dataset_proportion(dataset, sharp_ratio=0.9, blurry_ratio=0.1):
    sharp_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
    blurry_indices = [i for i, label in enumerate(dataset.labels) if label == 1]

    total_count = len(dataset)
    sharp_count = len(sharp_indices)
    blurry_count = len(blurry_indices)

    sharp_sample_size = int(total_count * sharp_ratio)
    blurry_sample_size = int(total_count * blurry_ratio)

    # 确保样本数量不会超过实际数量
    if sharp_sample_size > sharp_count:
        sharp_sample_size = sharp_count
        blurry_sample_size = total_count - sharp_sample_size
    elif blurry_sample_size > blurry_count:
        blurry_sample_size = blurry_count
        sharp_sample_size = total_count - blurry_sample_size

    sampled_sharp_indices = random.sample(sharp_indices, sharp_sample_size)
    sampled_blurry_indices = random.sample(blurry_indices, blurry_sample_size)

    sampled_indices = sampled_sharp_indices + sampled_blurry_indices
    random.shuffle(sampled_indices)

    return Subset(dataset, sampled_indices)

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


def train(model, train_loader, test_loader, checkpoint_dir,logger, num_epochs, class_weights):

    criterion = torch.nn.CrossEntropyLoss(weight = class_weights)
    optimizer = torch.optim.Adam([
        {'params': model.features.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-4)
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
        evaluate(model, test_loader,logger)

        scheduler.step(epoch_loss)


def evaluate(model, test_loader,logger):
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


    f1 = f1_score(all_labels, all_predictions)
    logger.info(f'F1 Score: {f1:.4f}')

    correct = sum(p == l for p, l in zip(all_predictions, all_labels))
    total = len(all_labels)
    accuracy = 100 * correct / total
    logger.info(f'Accuracy of the model on the test images: {accuracy:.2f}%')

def main():
    parser = argparse.ArgumentParser(description="Blurry Image Classifier Training")
    parser.add_argument('--epoch', type=int, required=True, help='Num of epoch of training process')
    args = parser.parse_args()
    logger = setup_logging()
    logger.info("Training started")

    dataset = BlurDataset(root_dir=ROOT_DIR, transform=transform)
    adjusted_dataset = adjust_dataset_proportion(dataset, sharp_ratio=0.9, blurry_ratio=0.1)

    # Calculate class weights
    class_counts = Counter(dataset.labels)
    total_count = sum(class_counts.values())
    class_weights = {cls: total_count / count for cls, count in class_counts.items()}
    class_weights = [class_weights[cls] for cls in range(len(class_counts))]
    class_weights = torch.FloatTensor(class_weights).to(device)

    train_size = int(0.8 * len(adjusted_dataset))
    test_size = len(adjusted_dataset) - train_size
    train_dataset, test_dataset = random_split(adjusted_dataset, [train_size, test_size])

    trian_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)
    
    from model import BCNN
    bcnn = BCNN(device=device)
    model = bcnn.get_model()

    # bcnn.unfreeze_layers(20)  # 仅解冻前20层进行微调
    train(model, trian_loader,test_loader, CHECKPOINT_DIR,logger,num_epochs=args.epoch, class_weights=class_weights)
    evaluate(model, test_loader,logger)
    logger.info("Training completed")


if __name__ == "__main__":
    main()
