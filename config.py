'''
Author: SheathedSharp z404878860@163.com
Date: 2025-01-12 12:35:46
'''
import os

class Config:
    # 基础路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'app', 'trained_models')

    # 确保模型目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 模型配置
    MODEL_CONFIGS = {
        'cnn_c16c32_k3_fc10': {
            'path': os.path.join(MODEL_DIR, 'cnn_c16c32_k3_fc10.pth'),
            'epochs': 5,
            'batch_size': 64,
            'learning_rate': 0.001
        },
        'cnn_c32c64_k5_fc512': {
            'path': os.path.join(MODEL_DIR, 'cnn_c32c64_k5_fc512.pth'),
            'epochs': 5,
            'batch_size': 64,
            'learning_rate': 0.001
        },
        'cnn_c32c64c128_k3_fc256': {
            'path': os.path.join(MODEL_DIR, 'cnn_c32c64c128_k3_fc256.pth'),
            'epochs': 5,
            'batch_size': 64,
            'learning_rate': 0.001
        },
        'mlp_512_256_128': {
            'path': os.path.join(MODEL_DIR, 'mlp_512_256_128.pth'),
            'epochs': 5,
            'batch_size': 64,
            'learning_rate': 0.001
        }
    }