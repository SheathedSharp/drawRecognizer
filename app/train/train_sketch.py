'''
Author: SheathedSharp z404878860@163.com
Date: 2025-01-15 03:40:00
'''
import os
import random
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from app.models.sketch_architectures import SketchResNet50, TripletLoss
from config import Config

class SketchyDataset(Dataset):
    """Sketchy Database 数据集加载器"""
    def __init__(self, sketch_dir, photo_dir, transform=None):
        self.sketch_dir = sketch_dir
        self.photo_dir = photo_dir
        self.transform = transform
        
        # 获取所有类别
        self.classes = [d for d in os.listdir(sketch_dir) 
                       if os.path.isdir(os.path.join(sketch_dir, d))]
        
        # 构建图像路径对
        self.pairs = []
        for cls in self.classes:
            sketch_cls_dir = os.path.join(sketch_dir, cls)
            photo_cls_dir = os.path.join(photo_dir, cls)
            
            sketches = [f for f in os.listdir(sketch_cls_dir) if f.endswith('.png')]
            photos = [f for f in os.listdir(photo_cls_dir) if f.endswith('.jpg')]
            
            for sketch in sketches:
                sketch_path = os.path.join(sketch_cls_dir, sketch)
                # 为每个草图找到对应的照片
                photo_path = os.path.join(photo_cls_dir, photos[0])  # 简单起见，使用第一张照片
                self.pairs.append((sketch_path, photo_path, cls))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        sketch_path, photo_path, cls = self.pairs[idx]
        
        # 加载草图和照片
        sketch = Image.open(sketch_path).convert('RGB')
        photo = Image.open(photo_path).convert('RGB')
        
        # 随机选择一个不同类别的负样本
        neg_cls = random.choice([c for c in self.classes if c != cls])
        neg_dir = os.path.join(self.photo_dir, neg_cls)
        neg_photos = os.listdir(neg_dir)
        neg_path = os.path.join(neg_dir, random.choice(neg_photos))
        negative = Image.open(neg_path).convert('RGB')
        
        if self.transform:
            sketch = self.transform(sketch)
            photo = self.transform(photo)
            negative = self.transform(negative)
            
        return sketch, photo, negative

def create_data_loader():
    """创建数据加载器"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = SketchyDataset(Config.SKETCHY_SKETCH_DIR, Config.SKETCHY_PHOTO_DIR, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.SKETCHY_BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    return train_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
        optimizer.zero_grad()
        
        # 提取特征
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)
        
        # 计算损失
        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f'Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader)

def print_gpu_utilization():
    """打印GPU使用情况"""
    import nvidia_smi
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def train_sketchy_model():
    """训练主函数"""
    device = torch.device(Config.DEVICE)
    model = SketchResNet50(feature_dim=Config.SKETCHY_FEATURE_DIM).to(device)
    
    # 创建 scaler 用于混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    train_loader = create_data_loader()
    criterion = TripletLoss(margin=Config.TRIPLET_MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    for epoch in range(Config.SKETCHY_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (sketch, photo, negative) in enumerate(train_loader):
            sketch = sketch.to(device)
            photo = photo.to(device)
            negative = negative.to(device)
            
            optimizer.zero_grad()
            
            # 使用自动混合精度
            with torch.cuda.amp.autocast():
                sketch_features = model(sketch)
                photo_features = model(photo)
                negative_features = model(negative)
                loss = criterion(sketch_features, photo_features, negative_features)
            
            # 使用 scaler 进行反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{Config.SKETCHY_EPOCHS}] '
                      f'Batch [{batch_idx+1}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f}')
            
            if (batch_idx + 1) % 50 == 0:
                print_gpu_utilization()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{Config.SKETCHY_EPOCHS}] Average Loss: {avg_loss:.4f}')
        
        # 保存模型
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), 
                      os.path.join(Config.SKETCH_MODEL_DIR, f'sketch_model_epoch_{epoch+1}.pth'))

if __name__ == '__main__':
    train_sketchy_model()