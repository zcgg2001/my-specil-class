"""
多任务模型: 品牌分类 + 香料含量回归

共享特征提取backbone，两个任务头
"""
try:
    from src.utils.safe_import import torch, nn
except ImportError:
    import torch
    import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_WAVELENGTHS, BRANDS


class SharedBackbone(nn.Module):
    """共享1D-CNN特征提取器"""
    
    def __init__(self, input_dim: int = NUM_WAVELENGTHS):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1: (B, 1, 313) -> (B, 64, 156)
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Block 2: (B, 64, 156) -> (B, 128, 78)
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Block 3: (B, 128, 78) -> (B, 256, 39)
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) 或 (B, 1, D)
            
        Returns:
            features: (B, 128)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        return self.fc(x)


class ClassificationHead(nn.Module):
    """品牌分类头"""
    
    def __init__(self, input_dim: int = 128, num_classes: int = len(BRANDS)):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class RegressionHead(nn.Module):
    """香料含量回归头"""
    
    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class MultiTaskModel(nn.Module):
    """多任务模型
    
    同时进行:
    1. 品牌分类 (二分类: hongmei vs yeshu)
    2. 香料含量回归 (0% - 3.6%)
    """
    
    def __init__(self, 
                 input_dim: int = NUM_WAVELENGTHS,
                 num_classes: int = len(BRANDS)):
        super().__init__()
        
        self.backbone = SharedBackbone(input_dim)
        self.cls_head = ClassificationHead(128, num_classes)
        self.reg_head = RegressionHead(128)
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, D) 光谱
            
        Returns:
            cls_logits: (B, num_classes) 分类logits
            reg_pred: (B, 1) 回归预测
        """
        features = self.backbone(x)
        cls_logits = self.cls_head(features)
        reg_pred = self.reg_head(features)
        return cls_logits, reg_pred
    
    def predict_brand(self, x: torch.Tensor) -> torch.Tensor:
        """仅预测品牌"""
        features = self.backbone(x)
        return self.cls_head(features)
    
    def predict_spice(self, x: torch.Tensor) -> torch.Tensor:
        """仅预测香料含量"""
        features = self.backbone(x)
        return self.reg_head(features)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MultiTaskModel().to(device)
    print(f"多任务模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(32, NUM_WAVELENGTHS).to(device)
    cls_logits, reg_pred = model(x)
    
    print(f"分类输出: {cls_logits.shape}")  # (32, 2)
    print(f"回归输出: {reg_pred.shape}")    # (32, 1)
