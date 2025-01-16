'''
Author: SheathedSharp z404878860@163.com
Date: 2025-01-12 12:35:46
'''
import os
import torch

class Config:
    # 基础路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'app', 'trained_models')
    GALLERY_DIR = os.path.join(BASE_DIR, 'app', 'static', 'gallery')
    DIGIT_MODEL_DIR = os.path.join(MODEL_DIR, 'digit_models')
    SKETCH_MODEL_DIR = os.path.join(MODEL_DIR, 'sketch_models')


    # 通用配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 手写数字训练参数
    DIGIT_EPOCHS = 20
    DIGIT_BATCH_SIZE = 128
    DIGIT_LEARNING_RATE = 1e-4

    # 手写数字模型配置
    DIGIT_MODEL_CONFIGS = {
        'cnn_c16c32_k3_fc10': {
            'path': os.path.join(DIGIT_MODEL_DIR, 'cnn_c16c32_k3_fc10.pth'),
        },
        'cnn_c32c64_k5_fc512': {
            'path': os.path.join(DIGIT_MODEL_DIR, 'cnn_c32c64_k5_fc512.pth'),
        },
        'cnn_c32c64c128_k3_fc256': {
            'path': os.path.join(DIGIT_MODEL_DIR, 'cnn_c32c64c128_k3_fc256.pth'),
        },
        'mlp_512_256_128': {
            'path': os.path.join(DIGIT_MODEL_DIR, 'mlp_512_256_128.pth'),
        }
    }