import torch
import mmcv
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, init_detector
import numpy as np
import albumentations as A
import mmcv.image.rotate as mmr
from PIL import Image
import cv2

# Define model
config_file = 'configs/retinanet/retinanet_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Define augmentations
train_transform = A.Compose([
    A.Rotate(limit=(-30, 30), border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=1.0),
    A.HorizontalFlip(),
    A.Normalize(),
    A.pytorch.ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(),
    A.pytorch.ToTensorV2()
])

# Define dataset class
class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_path, annos = self.data_list[idx]

        image = mmcv.imread(image_path)

        # Rotate the image and annotations
        if self.transform:
            angle = self.transform['angle']
            image, annos = mmr.imrotate(image, annos, angle)

        # Convert annotations to rotated bounding boxes
        bboxes = []
        labels = []
        for anno in annos:
            bbox = anno['bbox']
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
            points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            center, size, angle = cv2.minAreaRect(np.array(points))
            bbox = (center[0], center[1], size[0], size[1], angle)
            bboxes.append(bbox)
            labels.append(anno['category_id'])

        # Apply albumentations transform
        if self.transform:
            augmented = self.transform(image=image, bboxes=bboxes, category_id=labels)
            image = augmented["image"]
            bboxes = augmented["bboxes"]
            labels = augmented["category_id"]

        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        bboxes_tensor = torch.from_numpy(np.array(bboxes)).float()
        labels_tensor = torch.from_numpy(np.array(labels)).long()

        return image_tensor, bboxes_tensor, labels_tensor

# Define paths to data
train_data = [('train/image/1.jpg', [{'bbox': [100, 100, 200, 100, 200, 200, 100, 200], 'category_id': 1}]), ...]
val_data = [('val/image/1.jpg', [{'bbox': [100, 100, 200, 100, 200, 200, 100, 200], 'category_id': 1}]), ...]

# Define datasets and dataloaders
train_dataset = DetectionDataset(train_data, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate)

val_dataset = DetectionDataset(val_data, transform=val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate)
Define loss function and optimizer

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
Train the model

for epoch in range(10):
# Train for one epoch
  model.train()
  train_loss = 0
  for images, targets in train_loader:
    images = images.cuda()
    targets = targets.cuda()
    outputs = model(images)
    loss = criterion(outputs, targets)
    train_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

train_loss /= len(train_loader)
print(f"Epoch {epoch}: Training Loss = {train_loss}")

# Evaluate on validation set
model.eval()
val_loss = 0
for images, targets in val_loader:
    images = images.cuda()
    targets = targets.cuda()

    with torch.no_grad():
        outputs = model(images)
        loss = criterion(outputs, targets)
        val_loss += loss.item()

val_loss /= len(val_loader)
print(f"Epoch {epoch}: Validation Loss = {val_loss}")

# Save the trained model

torch.save(model.state_dict(), 'model.pth')
