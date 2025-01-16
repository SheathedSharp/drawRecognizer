'''
Author: SheathedSharp z404878860@163.com
Date: 2025-01-15 03:38:34
'''
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SketchBaseModel(nn.Module):
    """基础草图模型类"""
    def __init__(self, backbone='resnet50', feature_dim=512, pretrained=True):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 选择backbone
        if backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            self.in_features = 2048
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            self.in_features = 512
        elif backbone == 'vgg16':
            base_model = models.vgg16(pretrained=pretrained)
            self.in_features = 4096
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # 移除原始分类层
        if backbone.startswith('resnet'):
            self.features = nn.Sequential(*list(base_model.children())[:-1])
        elif backbone.startswith('vgg'):
            self.features = base_model.features
            
        # 添加新的特征映射层
        self.fc = nn.Sequential(
            nn.Linear(self.in_features, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, x):
        x = self.features(x)
        if len(x.shape) > 2:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
        x = self.fc(x)
        # L2标准化
        x = F.normalize(x, p=2, dim=1)
        return x

class SketchResNet50(SketchBaseModel):
    """ResNet50 based 草图检索模型"""
    def __init__(self, feature_dim=512):
        super().__init__(backbone='resnet50', feature_dim=feature_dim)

class SketchResNet34(SketchBaseModel):
    """ResNet34 based 草图检索模型"""
    def __init__(self, feature_dim=512):
        super().__init__(backbone='resnet34', feature_dim=feature_dim)

class SketchVGG16(SketchBaseModel):
    """VGG16 based 草图检索模型"""
    def __init__(self, feature_dim=512):
        super().__init__(backbone='vgg16', feature_dim=feature_dim)

class TripletLoss(nn.Module):
    """三元组损失函数"""
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        # 计算正样本对之间的距离
        dist_pos = torch.sum((anchor - positive) ** 2, dim=1)
        # 计算负样本对之间的距离
        dist_neg = torch.sum((anchor - negative) ** 2, dim=1)
        # 计算三元组损失
        losses = torch.relu(dist_pos - dist_neg + self.margin)
        return losses.mean()