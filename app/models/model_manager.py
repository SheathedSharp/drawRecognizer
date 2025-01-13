'''
Author: SheathedSharp z404878860@163.com
Date: 2025-01-12 18:41:14
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .model_architectures import *
from config import Config
import json
import os

class ModelManager:
    def __init__(self):
        self.models = {
            'cnn_c16c32_k3_fc10': DigitCNN_C16C32_K3_FC10,
            'cnn_c32c64_k5_fc512': DigitCNN_C32C64_K5_FC512,
            'cnn_c32c64c128_k3_fc256': DigitCNN_C32C64C128_K3_FC256,
            'mlp_512_256_128': DigitMLP_512_256_128
        }
        self.model_descriptions = {
            'cnn_c16c32_k3_fc10': {
                'type': 'CNN',
                'description': '轻量级双层CNN，小卷积核'
            },
            'cnn_c32c64_k5_fc512': {
                'type': 'CNN',
                'description': '中等规模双层CNN，大卷积核'
            },
            'cnn_c32c64c128_k3_fc256': {
                'type': 'CNN',
                'description': '深层三层CNN，渐进式通道增长'
            },
            'mlp_512_256_128': {
                'type': 'MLP',
                'description': '纯全连接网络，逐层降维'
            }
        }
        # 加载训练统计信息
        stats_path = os.path.join(Config.MODEL_DIR, 'training_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.model_stats = json.load(f)
        else:
            self.model_stats = {}
        self.current_model = None
        self.configs = Config.MODEL_CONFIGS

    def count_parameters(self, model):
        """计算模型的参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_model_info(self, model_name):
        """获取模型的详细信息"""
        if model_name not in self.models:
            return None

        model = self.models[model_name]()
        params = self.count_parameters(model)
        stats = self.model_stats.get(model_name, {})

        info = {
            'description': self.model_descriptions[model_name]['description'],
            'type': self.model_descriptions[model_name]['type'],
            'parameters': params,
            'stats': stats
        }
        return info

    def train_model(self, model_name, model, save_path, epochs=5, batch_size=64, learning_rate=0.001):
        device = next(model.parameters()).device
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 加载数据集
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, transform=transform, download=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        best_acc = 0.0
        stats = {
            'parameters': self.count_parameters(model),
            'training_history': []
        }
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if (i + 1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], '
                          f'Loss: {running_loss/100:.4f}, '
                          f'Accuracy: {100 * correct/total:.2f}%')
                    running_loss = 0.0
            
            # 验证阶段
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            
            test_acc = 100 * test_correct / test_total
            print(f'Validation Accuracy: {test_acc:.2f}%')
            
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), save_path)
                print(f'Model saved with accuracy: {best_acc:.2f}%')
            
            stats['training_history'].append({
                'epoch': epoch + 1,
                'train_accuracy': 100 * correct / total,
                'test_accuracy': test_acc
            })
        
        stats['final_accuracy'] = test_acc
        stats['best_accuracy'] = best_acc
        
        return stats
