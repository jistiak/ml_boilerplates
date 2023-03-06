import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import albumentations as A
import cv2
from PIL import Image
import segmentation_models_pytorch as smp

# Define dataset class
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image_tensor = transforms.functional.to_tensor(image)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor

# Define augmentations
train_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(),
    A.Rotate(10),
    A.Normalize(),
    A.pytorch.ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(),
    A.pytorch.ToTensorV2()
])

# Define paths to data
train_image_paths = ["train/image/1.png", "train/image/2.png", ...]
train_mask_paths = ["train/mask/1.png", "train/mask/2.png", ...]
val_image_paths = ["val/image/1.png", "val/image/2.png", ...]
val_mask_paths = ["val/mask/1.png", "val/mask/2.png", ...]

# Define datasets and dataloaders
train_dataset = SegmentationDataset(train_image_paths, train_mask_paths, transform=train_transform)
val_dataset = SegmentationDataset(val_image_paths, val_mask_paths, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# Define model
model = smp.FPN(
    encoder_name="resnet50",        # encoder architecture
    encoder_weights="imagenet",     # encoder weights
    classes=5,                      # number of classes
)

# Define loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    model.train()

    for i, (images, masks) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        for images, masks in val_loader:
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}, train_loss: {loss.item():.4f}, val_loss: {val_loss/len(val_loader):.4f}")
