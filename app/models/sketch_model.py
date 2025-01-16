'''
Author: SheathedSharp z404878860@163.com
Date: 2025-01-12 14:52:14
'''
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import base64
import io
import numpy as np

class SketchRetriever:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 使用预训练的ResNet50
        self.model = models.resnet50(pretrained=True)
        # 修改最后一层，输出512维特征
        self.model.fc = nn.Linear(2048, 512)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
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
    def build_gallery(self, image_dir):
        """构建图库特征"""
        self.gallery_paths = []
        features_list = []
        
        for img_path in sorted(glob.glob(os.path.join(image_dir, '*.jpg'))):
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            features = self.extract_features(image_tensor)
            
            features_list.append(features)
            self.gallery_paths.append(img_path)
        
        self.gallery_features = torch.cat(features_list, dim=0)
    
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
            similarity = similarities[0][idx].item()
            results.append({
                'path': img_path,
                'similarity': f"{similarity:.2f}"
            })
        
        return results