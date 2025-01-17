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
import numpy as np

class SketchyDataset(Dataset):
    """Sketchy Database 数据集加载器"""
    def __init__(self, sketch_dir, photo_dir, split='train', transform=None):
        self.sketch_dir = sketch_dir
        self.photo_dir = photo_dir
        self.transform = transform
        self.split = split
        
        # 获取所有类别
        self.classes = [d for d in os.listdir(sketch_dir) 
                       if os.path.isdir(os.path.join(sketch_dir, d))]
        
        print("classes:", self.classes)
        
        # 划分训练集和测试集类别 (80% 训练, 20% 测试)
        np.random.seed(42)  # 固定随机种子
        n_classes = len(self.classes)
        n_train = int(0.8 * n_classes)
        
        if split == 'train':
            self.used_classes = self.classes[:n_train]
        else:  # test
            self.used_classes = self.classes[n_train:]
            
        # 构建图像路径对
        self.pairs = []
        for cls in self.used_classes:
            sketch_cls_dir = os.path.join(sketch_dir, cls)
            photo_cls_dir = os.path.join(photo_dir, cls)
            
            sketches = [f for f in os.listdir(sketch_cls_dir) if f.endswith('.png')]
            photos = [f for f in os.listdir(photo_cls_dir) if f.endswith('.jpg')]
            
            # 对每个类别内的数据也进行训练/测试划分
            n_sketches = len(sketches)
            n_train_sketches = int(0.8 * n_sketches)
            
            if split == 'train':
                used_sketches = sketches[:n_train_sketches]
            else:
                used_sketches = sketches[n_train_sketches:]
            
            for sketch in used_sketches:
                sketch_path = os.path.join(sketch_cls_dir, sketch)
                # 随机选择一张对应类别的照片
                photo_path = os.path.join(photo_cls_dir, random.choice(photos))
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

def create_data_loaders():
    """创建训练集和测试集数据加载器"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建训练集
    train_dataset = SketchyDataset(
        Config.SKETCHY_SKETCH_DIR, 
        Config.SKETCHY_PHOTO_DIR,
        split='train',
        transform=transform
    )
    
    # 创建测试集
    test_dataset = SketchyDataset(
        Config.SKETCHY_SKETCH_DIR,
        Config.SKETCHY_PHOTO_DIR,
        split='test', 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.SKETCHY_BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.SKETCHY_BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, test_loader

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

def validate(model, test_loader, criterion, device):
    """验证函数"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, (sketch, photo, negative) in enumerate(test_loader):
            sketch = sketch.to(device)
            photo = photo.to(device)
            negative = negative.to(device)
            
            sketch_features = model(sketch)
            photo_features = model(photo)
            negative_features = model(negative)
            
            loss = criterion(sketch_features, photo_features, negative_features)
            total_loss += loss.item()
            
    return total_loss / len(test_loader)

def train_sketchy_model():
    """训练主函数"""
    device = torch.device(Config.DEVICE)
    model = SketchResNet50(feature_dim=Config.SKETCHY_FEATURE_DIM).to(device)
    
    scaler = torch.cuda.amp.GradScaler()
    
    # 获取训练集和测试集加载器
    train_loader, test_loader = create_data_loaders()
    criterion = TripletLoss(margin=Config.SKETCHY_TRIPLET_MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=Config.SKETCHY_LEARNING_RATE)
    
    stats = {
        'training_time': 0,
        'best_loss': float('inf'),
        'best_test_loss': float('inf'),
        'train_losses': [],
        'test_losses': []
    }
    
    start_time = time.time()
    patience = 10
    no_improve = 0
    
    for epoch in range(Config.SKETCHY_EPOCHS):
        # 训练阶段
        model.train()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 测试阶段
        model.eval()
        test_loss = validate(model, test_loader, criterion, device)
        
        print(f'Epoch [{epoch+1}/{Config.SKETCHY_EPOCHS}]')
        print(f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        
        stats['train_losses'].append(train_loss)
        stats['test_losses'].append(test_loss)
        
        # 基于测试集损失保存最佳模型
        if test_loss < stats['best_test_loss']:
            stats['best_test_loss'] = test_loss
            torch.save(model.state_dict(), 
                      os.path.join(Config.SKETCH_MODEL_DIR, 'sketch_model_best.pth'))
            print(f'Model saved with test loss: {test_loss:.4f}')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping after {patience} epochs without improvement')
                break
        
        # 每个epoch都保存一个检查点
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'best_loss': stats['best_loss']
        }, os.path.join(Config.SKETCH_MODEL_DIR, f'sketch_model_epoch_{epoch+1}.pth'))
    
    stats['training_time'] = time.time() - start_time
    stats['best_loss'] = stats['best_test_loss']
    
    # 保存训练统计信息
    stats_path = os.path.join(Config.SKETCH_MODEL_DIR, 'sketch_training_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print('Training completed!')

if __name__ == '__main__':
    train_sketchy_model()