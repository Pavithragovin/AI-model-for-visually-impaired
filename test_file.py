import os
import torch
from torch.utils.data import DataLoader
from cnn_model import StreetAidCNN
from loss import my_loss
from helpers import direction_performance
import dataset
import importlib
from torchvision import transforms

# ==== Device ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== Paths ====
test_csv = '/content/drive/MyDrive/training_file.csv'#✅ Your test CSV
test_img_dir = '/content/drive/MyDrive/PTL_Dataset_768x576/PTL_Dataset_768x576'  # ✅ Folder with test images
model_path = '/content/drive/MyDrive/model_epoch_70.pt'  # ✅ Trained model path

# ==== Transform ====
data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ==== Load Dataset ====
importlib.reload(dataset)
TrafficLightDataset = dataset.TrafficLightDataset

test_dataset = TrafficLightDataset(test_csv, test_img_dir,transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ==== Load Model ====
model = StreetAidCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ==== Evaluation ====
total_loss = 0.0
total_correct = 0
total_samples = 0
total_angle_err = 0.0
total_start_err = 0.0
total_end_err = 0.0
processed_batches = 0

with torch.no_grad():
    for batch in test_loader:
        try:
            images = batch['image'].to(device)
            labels = batch['mode'].to(device)
            points = batch['points'].to(device)

            pred_cls, pred_coords = model(images)
            loss, _, _ = my_loss(pred_cls, pred_coords, points, labels)

            total_loss += loss.item()
            total_correct += (pred_cls.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

            angle_err, start_err, end_err = direction_performance(pred_coords, points)
            total_angle_err += angle_err
            total_start_err += start_err
            total_end_err += end_err

            processed_batches += 1

        except Exception as e:
            print(f"Skipping a batch due to error: {e}")

# ==== Final Summary ====
if total_samples > 0 and processed_batches > 0:
    avg_loss = total_loss / processed_batches
    accuracy = 100 * total_correct / total_samples
    avg_angle = total_angle_err / processed_batches
    avg_start = total_start_err / processed_batches
    avg_end = total_end_err / processed_batches

    print("\n📊 === Test Evaluation Summary ===")
    print(f"✅ Accuracy          : {accuracy:.2f}%")
    print(f"📉 Average Loss      : {avg_loss:.4f}")
    print(f"📐 Avg Angle Error   : {avg_angle:.2f}")
    print(f"📍 Avg Start Error   : {avg_start:.2f}")
    print(f"📍 Avg End Error     : {avg_end:.2f}")
    print(f"📦 Processed {processed_batches} batches out of {len(test_loader)}")
else:
    print("❌ No test samples processed. Please check test_csv, image paths, or dataset loading.")
