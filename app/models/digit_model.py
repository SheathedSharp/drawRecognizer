import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import io
import base64
from .model_architectures import *

class DigitRecognizer:
    def __init__(self, model_name='cnn_c16c32_k3_fc10', model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 根据模型名称选择对应的模型架构
        model_classes = {
            'cnn_c16c32_k3_fc10': DigitCNN_C16C32_K3_FC10,
            'cnn_c32c64_k5_fc512': DigitCNN_C32C64_K5_FC512,
            'cnn_c32c64c128_k3_fc256': DigitCNN_C32C64C128_K3_FC256,
            'mlp_512_256_128': DigitMLP_512_256_128
        }
        
        self.model = model_classes[model_name]().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def train(self, epochs=10, batch_size=64, save_path='digit_model.pth'):
        # 加载MNIST数据集
        train_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            transform=self.transform,
            download=True
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=False,
            transform=self.transform,
            download=True
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        best_acc = 0.0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
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

            # Validation phase
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(
                        self.device), labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            acc = 100 * correct / total
            print(f'Validation Accuracy: {acc:.2f}%')

            if acc > best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(), save_path)
                print(f'Model saved with accuracy: {best_acc:.2f}%')

        print('Training finished!')

    def preprocess_image(self, image_data):
        try:
            # 移除base64头部信息
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('L')
            
            # 打印原始图像的大小和像素范围
            print(f"Original image size: {image.size}")
            
            # 转换为numpy数组并反转颜色
            image_array = np.array(image)
            print(f"Image array range before inversion: {image_array.min()} to {image_array.max()}")
            
            # 确保黑白对比度
            threshold = 128
            image_array = np.where(image_array > threshold, 0, 255)
            
            # 转回Image对象
            image = Image.fromarray(image_array.astype('uint8'))
            
            # 应用转换
            image_tensor = self.transform(image)
            print(f"Tensor shape: {image_tensor.shape}")
            print(f"Tensor range: {image_tensor.min():.2f} to {image_tensor.max():.2f}")
            
            # 保存处理后的图像用于调试
            debug_path = 'debug_image.png'
            image.save(debug_path)
            print(f"Saved debug image to {debug_path}")
            
            return image_tensor.unsqueeze(0)
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            traceback.print_exc()
            return None

    def predict(self, image_data):
        self.model.eval()
        with torch.no_grad():
            try:
                image_tensor = self.preprocess_image(image_data)
                if image_tensor is None:
                    return None

                image_tensor = image_tensor.to(self.device)
                output = self.model(image_tensor)
                _, predicted = torch.max(output.data, 1)
                print("result:", predicted.item())
                return predicted.item()
            except Exception as e:
                print(f"Error during prediction: {e}")
                return None
