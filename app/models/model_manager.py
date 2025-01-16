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
from .digit_model import *
from config import Config
import json
import os

class DigitModelManager:
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
        stats_path = os.path.join(Config.DIGIT_MODEL_DIR, 'training_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.model_stats = json.load(f)
        else:
            self.model_stats = {}
        self.current_model = None
        self.configs = Config.DIGIT_MODEL_CONFIGS

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
    
    def check_models_status(self):
        """检查所有模型的状态并返回状态信息"""
        status = {}
        for model_name, config in Config.DIGIT_MODEL_CONFIGS.items():
            model_path = config['path']
            status[model_name] = {
                'exists': os.path.exists(model_path),
                'path': model_path,
                'description': self.model_descriptions[model_name]['description'],
                'type': self.model_descriptions[model_name]['type']
            }
            
            # 如果存在训练统计信息，添加到状态中
            if model_name in self.model_stats:
                status[model_name].update({
                    'accuracy': self.model_stats[model_name].get('best_accuracy', 'Unknown'),
                    'parameters': self.model_stats[model_name].get('parameters', 'Unknown')
                })
        
        return status
