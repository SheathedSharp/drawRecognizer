'''
Author: SheathedSharp z404878860@163.com
Date: 2025-01-15 03:47:05
'''
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from app.models.digit_architectures import *
from app.models.model_manager import DigitModelManager
from config import Config

def create_data_loader():
    """创建数据加载器"""
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform, download=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.DIGIT_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.DIGIT_BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        log_probs = nn.functional.log_softmax(outputs, dim=1)
        loss = nn.functional.nll_loss(log_probs, labels)
            
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'Batch [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Accuracy: {100 * correct/total:.2f}%')
    
    return total_loss / len(train_loader), 100 * correct / total

def validate(model, test_loader, criterion, device):
    """验证模型性能"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            log_probs = nn.functional.log_softmax(outputs, dim=1)
            loss = nn.functional.nll_loss(log_probs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / len(test_loader), 100 * correct / total

def train_digit_models(model_names=None):
    """
    训练主函数
    Args:
        model_names: 需要训练的模型名称列表，如果为None则训练所有模型
    """
    device = torch.device(Config.DEVICE)
    manager = DigitModelManager()
    training_stats = {}
    
    # 加载已存在的训练统计信息（如果有）
    stats_path = os.path.join(Config.DIGIT_MODEL_DIR, 'digit_training_stats.json')
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            training_stats = json.load(f)
    
    train_loader, test_loader = create_data_loader()
    
    # 如果没有指定模型名称，则训练所有模型
    models_to_train = model_names if model_names else Config.DIGIT_MODEL_CONFIGS.keys()
    
    for model_name in models_to_train:
        if model_name not in Config.DIGIT_MODEL_CONFIGS:
            print(f"警告: 未知的模型 {model_name}，跳过训练")
            continue
            
        print(f"\nTraining digit model: {model_name}...")
        
        model = manager.models[model_name]().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=Config.DIGIT_LEARNING_RATE, 
            weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        stats = {
            'training_time': 0,
            'best_accuracy': 0,
            'final_accuracy': 0,
            'parameters': sum(p.numel() for p in model.parameters())
        }
        
        start_time = time.time()
        
        patience = 10
        no_improve = 0
        
        for epoch in range(Config.DIGIT_EPOCHS):
            print(f'\nEpoch {epoch+1}/{Config.DIGIT_EPOCHS}')
            
            # 训练阶段
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device)
            
            # 验证阶段
            val_loss, val_acc = validate(model, test_loader, criterion, device)
            scheduler.step(val_acc)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            if val_acc > stats['best_accuracy']:
                stats['best_accuracy'] = val_acc
                torch.save(model.state_dict(), Config.DIGIT_MODEL_CONFIGS[model_name]['path'])
                print(f'Model saved with accuracy: {val_acc:.2f}%')
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f'Early stopping after {patience} epochs without improvement')
                    break
        
        stats['training_time'] = time.time() - start_time
        stats['final_accuracy'] = val_acc
        
        training_stats[model_name] = {
            'training_time': stats['training_time'],
            'final_accuracy': stats['final_accuracy'],
            'best_accuracy': stats['best_accuracy'],
            'parameters': stats['parameters'],
            'epochs': Config.DIGIT_EPOCHS,
            'batch_size': Config.DIGIT_BATCH_SIZE
        }
    
    # 保存训练统计信息
    stats_path = os.path.join(Config.DIGIT_MODEL_DIR, 'digit_training_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=4)
    
    print('Training completed!')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train digit models')
    parser.add_argument('--models', nargs='+', help='Model names to train, e.g.: cnn_c16c32_k3_fc10 mlp_512_256_128')
    parser.add_argument('--list', action='store_true', help='List all available models')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available models:")
        for model_name, config in Config.DIGIT_MODEL_CONFIGS.items():
            print(f"- {model_name}")
        exit(0)
    
    train_digit_models(args.models) 