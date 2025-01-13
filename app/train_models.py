'''
Author: SheathedSharp z404878860@163.com
Date: 2025-01-12 18:49:25
'''
import torch
from app.models.model_architectures import *
from app.models.model_manager import ModelManager
from config import Config
import json
import time
import os

def train_all_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manager = ModelManager()
    training_stats = {}

    for model_name, config in Config.MODEL_CONFIGS.items():
        print(f"\nTraining {model_name}...")
        model = manager.models[model_name]().to(device)
        
        # 记录开始时间
        start_time = time.time()
        
        # 训练模型
        stats = manager.train_model(
            model_name=model_name,
            model=model,
            save_path=config['path'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate']
        )
        
        # 记录训练时间
        training_time = time.time() - start_time
        
        # 保存训练统计信息
        training_stats[model_name] = {
            'training_time': training_time,
            'final_accuracy': stats['final_accuracy'],
            'best_accuracy': stats['best_accuracy'],
            'parameters': stats['parameters'],
            'epochs': config['epochs'],
            'batch_size': config['batch_size']
        }
    
    # 保存训练统计信息到文件
    stats_path = os.path.join(Config.MODEL_DIR, 'training_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=4)
    
    print("\nAll models trained successfully!")
    print(f"Training statistics saved to {stats_path}")

if __name__ == "__main__":
    train_all_models()