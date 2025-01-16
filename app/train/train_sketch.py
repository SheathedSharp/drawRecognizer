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

class SketchDataset(Dataset):
    """草图数据集加载器"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.image_paths = []
        
        # 构建三元组数据
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            images = os.listdir(class_dir)
            for img in images:
                self.image_paths.append((os.path.join(class_dir, img), class_name))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path, class_name = self.image_paths[idx]
        # 加载锚点图像
        anchor = Image.open(img_path).convert('RGB')
        if self.transform:
            anchor = self.transform(anchor)
            
        # 获取同类别的正样本
        pos_paths = [p for p, c in self.image_paths if c == class_name and p != img_path]
        pos_path = random.choice(pos_paths)
        positive = Image.open(pos_path).convert('RGB')
        if self.transform:
            positive = self.transform(positive)
            
        # 获取不同类别的负样本
        neg_paths = [p for p, c in self.image_paths if c != class_name]
        neg_path = random.choice(neg_paths)
        negative = Image.open(neg_path).convert('RGB')
        if self.transform:
            negative = self.transform(negative)
            
        return anchor, positive, negative

def create_data_loader():
    """创建数据加载器"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = SketchDataset(Config.SKETCH_TRAIN_DIR, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4
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

def train_sketch_model():
    """训练主函数"""
    device = torch.device(Config.DEVICE)
    model = SketchResNet50(feature_dim=Config.FEATURE_DIM).to(device)
    
    train_loader = create_data_loader()
    criterion = TripletLoss(margin=Config.TRIPLET_MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    stats = {
        'training_time': 0,
        'best_loss': float('inf'),
        'parameters': sum(p.numel() for p in model.parameters())
    }
    
    start_time = time.time()
    
    for epoch in range(Config.SKETCH_EPOCHS):
        print(f'\nEpoch {epoch+1}/{Config.SKETCH_EPOCHS}')
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        print(f'Average Loss: {avg_loss:.4f}')
        
        if avg_loss < stats['best_loss']:
            stats['best_loss'] = avg_loss
            torch.save(model.state_dict(), Config.SKETCH_MODEL_PATH)
            print(f'Model saved with loss: {avg_loss:.4f}')
    
    stats['training_time'] = time.time() - start_time
    
    # 保存训练统计信息
    stats_path = os.path.join(Config.SKETCH_MODEL_DIR, 'sketch_training_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print('Training completed!')

if __name__ == '__main__':
    train_sketch_model()