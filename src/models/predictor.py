"""
香料含量预测器

用于:
1. 作为独立回归模型预测香料含量
2. 在cVAE训练中提供L_attr约束
"""
try:
    from src.utils.safe_import import torch, nn
except ImportError:
    import torch
    import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_WAVELENGTHS


class SpicePredictor(nn.Module):
    """香料含量预测器 (1D-CNN)
    
    用于预测光谱对应的香料含量(连续值回归)
    """
    
    def __init__(self, input_dim: int = NUM_WAVELENGTHS):
        super().__init__()
        
        # 1D卷积特征提取
        self.conv_layers = nn.Sequential(
            # (B, 1, 313) -> (B, 32, 156)
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # (B, 32, 156) -> (B, 64, 78)
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # (B, 64, 78) -> (B, 128, 39)
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 全局平均池化 -> (B, 128, 1)
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) 或 (B, 1, D) 光谱
            
        Returns:
            pred: (B, 1) 预测的香料含量
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, D) -> (B, 1, D)
        
        features = self.conv_layers(x)  # (B, 128, 1)
        features = features.squeeze(-1)  # (B, 128)
        return self.regressor(features)


class SimpleMLP(nn.Module):
    """简单MLP预测器 (用于快速训练)"""
    
    def __init__(self, input_dim: int = NUM_WAVELENGTHS):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)  # (B, 1, D) -> (B, D)
        return self.net(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试CNN预测器
    model = SpicePredictor().to(device)
    print(f"CNN参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(32, NUM_WAVELENGTHS).to(device)
    pred = model(x)
    print(f"输入: {x.shape}, 输出: {pred.shape}")
    
    # 测试MLP预测器
    mlp = SimpleMLP().to(device)
    print(f"MLP参数量: {sum(p.numel() for p in mlp.parameters()):,}")
