'''
Author: SheathedSharp z404878860@163.com
Date: 2025-01-12 14:52:14
'''
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import base64
import io
import numpy as np
import os
from config import Config

class SketchRetriever:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 使用预训练的ResNet50
        self.model = models.resnet50(pretrained=True)
        # 修改最后一层，输出512维特征
        self.model.fc = nn.Linear(2048, 512)
        
        if model_path and os.path.exists(model_path):
            try:
                # 加载训练好的权重
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # 处理加载的状态字典
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # 创建新的状态字典，调整键名
                new_state_dict = {}
                for k, v in state_dict.items():
                    # 移除模块前缀（如果存在）
                    k = k.replace('module.', '')
                    if k.startswith('fc.'):
                        new_state_dict[k] = v
                    else:
                        new_state_dict[k] = v
                
                # 加载调整后的状态字典
                self.model.load_state_dict(new_state_dict, strict=False)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using pretrained model instead.")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # 存储图库特征
        self.gallery_features = None
        self.gallery_paths = []
        
    def preprocess_image(self, image_data):
        # 处理base64图像数据
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def extract_features(self, image_tensor):
        with torch.no_grad():
            features = self.model(image_tensor)
            return F.normalize(features, p=2, dim=1)
            
    def build_gallery(self, image_dir, samples_per_category=10):
        """构建精简版图库特征
        Args:
            image_dir: 图库目录路径
            samples_per_category: 每个类别选择的图片数量
        """
        self.gallery_paths = []
        features_list = []
        
        if not os.path.exists(image_dir):
            print(f"Gallery directory not found: {image_dir}")
            return
        
        # 获取所有类别
        categories = [d for d in os.listdir(image_dir) 
                    if os.path.isdir(os.path.join(image_dir, d))]
        
        print(f"Found {len(categories)} categories")
        
        # 批量处理图片
        batch_size = 32
        batch_tensors = []
        batch_paths = []
        
        for category in categories:
            category_path = os.path.join(image_dir, category)
            # 获取该类别下的所有图片
            images = [f for f in os.listdir(category_path) if f.endswith('.jpg')]
            
            # 随机选择指定数量的图片
            selected_images = random.sample(images, min(samples_per_category, len(images)))
            print(f"Category {category}: selected {len(selected_images)} images")
            
            for img_name in selected_images:
                img_path = os.path.join(category_path, img_name)
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = self.transform(image).unsqueeze(0)
                    batch_tensors.append(image_tensor)
                    batch_paths.append(img_path)
                    
                    # 当积累足够的批次时进行处理
                    if len(batch_tensors) >= batch_size:
                        batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
                        with torch.cuda.amp.autocast():
                            with torch.no_grad():
                                batch_features = self.extract_features(batch_tensor)
                        features_list.append(batch_features.cpu())
                        self.gallery_paths.extend(batch_paths)
                        
                        # 清空批次
                        batch_tensors = []
                        batch_paths = []
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        # 处理剩余的图片
        if batch_tensors:
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    batch_features = self.extract_features(batch_tensor)
            features_list.append(batch_features.cpu())
            self.gallery_paths.extend(batch_paths)
        
        # 合并所有特征
        if features_list:
            self.gallery_features = torch.cat(features_list, dim=0).to(self.device)
            print(f"\nGallery built successfully with {len(self.gallery_paths)} images")
            
            # 保存精简版特征缓存
            cache_file = os.path.join(Config.SKETCH_MODEL_DIR, 'gallery_features_small.pt')
            torch.save({
                'features': self.gallery_features.cpu(),
                'paths': self.gallery_paths
            }, cache_file)
    
    def retrieve(self, sketch_data, top_k=5):
        """检索相似图片"""
        # 处理输入草图
        sketch_tensor = self.preprocess_image(sketch_data)
        sketch_features = self.extract_features(sketch_tensor)
        
        # 计算相似度
        similarities = torch.mm(sketch_features, self.gallery_features.t())
        _, indices = similarities[0].topk(top_k)
        
        # 返回结果
        results = []
        for idx in indices:
            img_path = self.gallery_paths[idx]
            # 将完整路径转换为相对路径
            relative_path = os.path.relpath(img_path, Config.SKETCHY_PHOTO_DIR)
            # 替换反斜杠为正斜杠（对 Windows 系统很重要）
            relative_path = relative_path.replace('\\', '/')
            similarity = similarities[0][idx].item()
            results.append({
                'path': f'/gallery_image/{relative_path}',
                'similarity': f"{similarity:.2f}"
            })
        
        return results