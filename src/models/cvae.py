"""
物理约束条件变分自编码器 (Physics-Constrained cVAE)

用于生成具有物理合理性的近红外光谱数据
"""
try:
    from src.utils.safe_import import torch, nn
except ImportError:
    import torch
    import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LATENT_DIM, HIDDEN_DIMS, NUM_WAVELENGTHS


class Encoder(nn.Module):
    """cVAE编码器
    
    x ⊕ condition → μ, logσ²
    """
    
    def __init__(self, 
                 input_dim: int = NUM_WAVELENGTHS,
                 condition_dim: int = 3,
                 hidden_dims: list = HIDDEN_DIMS,
                 latent_dim: int = LATENT_DIM):
        super().__init__()
        
        # 输入维度: 光谱 + 条件
        in_features = input_dim + condition_dim
        
        layers = []
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_features, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            in_features = h_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # 输出均值和对数方差
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
    def forward(self, x: torch.Tensor, 
                condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim) 输入光谱
            condition: (B, condition_dim) 条件向量
            
        Returns:
            mu: (B, latent_dim) 均值
            logvar: (B, latent_dim) 对数方差
        """
        h = torch.cat([x, condition], dim=1)
        h = self.encoder(h)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """cVAE解码器
    
    z ⊕ condition → x_recon
    """
    
    def __init__(self,
                 output_dim: int = NUM_WAVELENGTHS,
                 condition_dim: int = 3,
                 hidden_dims: list = None,
                 latent_dim: int = LATENT_DIM):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = list(reversed(HIDDEN_DIMS))
        
        in_features = latent_dim + condition_dim
        
        layers = []
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_features, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            in_features = h_dim
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        # 输出使用Sigmoid确保[0,1]范围
        layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor, 
                condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) 隐变量
            condition: (B, condition_dim) 条件向量
            
        Returns:
            x_recon: (B, output_dim) 重建光谱
        """
        h = torch.cat([z, condition], dim=1)
        return self.decoder(h)


class PhysicsConstrainedCVAE(nn.Module):
    """物理约束条件变分自编码器
    
    特点:
    1. 条件生成: 以品牌和香料含量为条件
    2. 物理约束: 通过损失函数实现平滑性、范围约束
    3. 属性一致性: 生成光谱需通过预测器验证
    """
    
    def __init__(self,
                 input_dim: int = NUM_WAVELENGTHS,
                 condition_dim: int = 3,
                 hidden_dims: list = HIDDEN_DIMS,
                 latent_dim: int = LATENT_DIM):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        self.encoder = Encoder(input_dim, condition_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(input_dim, condition_dim, 
                               list(reversed(hidden_dims)), latent_dim)
        
    def reparameterize(self, mu: torch.Tensor, 
                       logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧
        
        z = μ + σ * ε, where ε ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def forward(self, x: torch.Tensor, 
                condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            x: (B, input_dim) 输入光谱
            condition: (B, condition_dim) 条件 [brand_onehot, spice_normalized]
            
        Returns:
            x_recon: (B, input_dim) 重建光谱
            mu: (B, latent_dim) 编码均值
            logvar: (B, latent_dim) 编码对数方差
        """
        mu, logvar = self.encoder(x, condition)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, condition)
        return x_recon, mu, logvar
    
    def encode(self, x: torch.Tensor, 
               condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码"""
        return self.encoder(x, condition)
    
    def decode(self, z: torch.Tensor, 
               condition: torch.Tensor) -> torch.Tensor:
        """解码"""
        return self.decoder(z, condition)
    
    def generate(self, condition: torch.Tensor, 
                 n_samples: int = 1) -> torch.Tensor:
        """根据条件生成新光谱
        
        Args:
            condition: (B, condition_dim) 或 (condition_dim,) 条件向量
            n_samples: 每个条件生成的样本数
            
        Returns:
            generated: (B*n_samples, input_dim) 生成的光谱
        """
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        
        batch_size = condition.size(0)
        device = next(self.parameters()).device
        
        # 扩展条件
        condition = condition.repeat_interleave(n_samples, dim=0)
        
        # 从先验采样
        z = torch.randn(batch_size * n_samples, self.latent_dim, device=device)
        
        # 解码
        with torch.no_grad():
            generated = self.decode(z, condition)
        
        return generated
    
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor,
                    condition: torch.Tensor, 
                    n_steps: int = 10) -> torch.Tensor:
        """在两个光谱之间的隐空间插值
        
        Args:
            x1, x2: (1, input_dim) 两个光谱
            condition: (1, condition_dim) 条件
            n_steps: 插值步数
            
        Returns:
            interpolated: (n_steps, input_dim)
        """
        mu1, _ = self.encode(x1, condition)
        mu2, _ = self.encode(x2, condition)
        
        # 线性插值
        alphas = torch.linspace(0, 1, n_steps, device=mu1.device)
        interpolated = []
        
        for alpha in alphas:
            z = (1 - alpha) * mu1 + alpha * mu2
            x = self.decode(z, condition)
            interpolated.append(x)
        
        return torch.cat(interpolated, dim=0)


if __name__ == "__main__":
    # 测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PhysicsConstrainedCVAE().to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    x = torch.randn(32, NUM_WAVELENGTHS).to(device)
    condition = torch.randn(32, 3).to(device)
    
    x_recon, mu, logvar = model(x, condition)
    print(f"输入形状: {x.shape}")
    print(f"重建形状: {x_recon.shape}")
    print(f"隐变量形状: {mu.shape}")
    
    # 测试生成
    generated = model.generate(condition[:1], n_samples=5)
    print(f"生成形状: {generated.shape}")
