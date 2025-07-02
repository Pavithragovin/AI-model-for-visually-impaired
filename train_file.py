#50 completed
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from cnn_model import StreetAidCNN
import dataset # Import the module
from loss import my_loss # Corrected import to my_loss
from helpers import direction_performance
import importlib # Import importlib
from torchvision import transforms # Import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
train_csv = 'training_file.csv'
val_csv = 'validation_file.csv'
# Corrected the image directory paths to include the nested folder
train_img_dir = '/content/drive/MyDrive/PTL_Dataset_768x576/PTL_Dataset_768x576'
val_img_dir = '/content/drive/MyDrive/PTL_Dataset_876x657/PTL_Dataset_876x657' # Assuming a similar nested structure for validation data
save_path = '/content/drive/MyDrive'
os.makedirs(save_path, exist_ok=True)

# Define transformations
# Include ToTensor() to convert PIL Image to PyTorch Tensor
data_transform = transforms.Compose([
    transforms.Resize((128, 128)), # Example resize, adjust as needed
    transforms.ToTensor()
])


# Explicitly reload the dataset module
importlib.reload(dataset)
TrafficLightDataset = dataset.TrafficLightDataset # Get the reloaded class


# Pass the transform to the datasets
train_dataset = TrafficLightDataset(train_csv, train_img_dir, transform=data_transform)
val_dataset = TrafficLightDataset(val_csv, val_img_dir, transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

model = StreetAidCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses, val_losses = [], []
train_accs, val_accs = [], []
val_angles, val_starts, val_ends = [], [], []

for epoch in range(1, 51):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in train_loader:
        # The batch should now contain tensors because of the transform
        images = batch['image'].to(device)
        labels = batch['mode'].to(device)
        points = batch['points'].to(device)

        pred_cls, pred_coords = model(images)
        loss, _, _ = my_loss(pred_cls, pred_coords, points, labels) # Corrected function call

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (pred_cls.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    angle_err, start_err, end_err = 0, 0, 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['mode'].to(device)
            points = batch['points'].to(device)

            pred_cls, pred_coords = model(images)
            loss, _, _ = my_loss(pred_cls, pred_coords, points, labels) # Corrected function call

            val_loss += loss.item()
            val_correct += (pred_cls.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

            angle, start, end = direction_performance(pred_coords, points)
            angle_err += angle
            start_err += start
            end_err += end


    # Check if val_total is not zero before calculating averages
    if val_total > 0:
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_angles.append(angle_err / len(val_loader))
        val_starts.append(start_err / len(val_loader))
        val_ends.append(end_err / len(val_loader))
    else:
        # Handle case where val_loader is empty or all batches are skipped
        print(f"Epoch {epoch:02d}: No validation batches processed.")
        val_losses.append(0) # Append dummy values
        val_accs.append(0)
        val_angles.append(0)
        val_starts.append(0)
        val_ends.append(0)


    print(f"Epoch {epoch:02d}: "
          f"TrainLoss={train_losses[-1]:.4f}, Acc={train_accs[-1]:.2f}% | "
          f"ValLoss={val_losses[-1]:.4f}, Acc={val_accs[-1]:.2f}% | "
          f"Angle={val_angles[-1]:.2f}, Start={val_starts[-1]:.2f}, End={val_ends[-1]:.2f}")

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"{save_path}/model_epoch_{epoch}.pt")

# Final save
torch.save(model.state_dict(), f"{save_path}/final_model.pt")

# Plots
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("Loss"); plt.legend(); plt.show()

plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.title("Accuracy"); plt.legend(); plt.show()

plt.plot(val_angles, label='Angle Err')
plt.plot(val_starts, label='Start Err')
plt.plot(val_ends, label='End Err')
plt.title("Zebra Midline Error"); plt.legend(); plt.show()