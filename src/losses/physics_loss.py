"""
物理约束损失函数模块

包含:
1. L_smooth: 二阶差分平滑约束
2. L_range: 输出范围[0,1]约束  
3. L_attr: 香料含量属性一致性约束
4. PhysicsConstrainedVAELoss: 完整损失函数
"""
try:
    from src.utils.safe_import import torch, nn
except ImportError:
    import torch
    import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def smooth_loss(x: torch.Tensor) -> torch.Tensor:
    """二阶差分平滑约束
    
    L_smooth = ||∇²x||² = ||(x[i+1] - 2*x[i] + x[i-1])||²
    
    鼓励生成光谱具有物理上合理的平滑性
    
    Args:
        x: (B, D) 光谱张量
        
    Returns:
        loss: 标量
    """
    # 二阶差分: d²x/di² ≈ x[i+1] - 2*x[i] + x[i-1]
    second_diff = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    return torch.mean(second_diff ** 2)


def range_loss(x: torch.Tensor, 
               min_val: float = 0.0, 
               max_val: float = 1.0) -> torch.Tensor:
    """输出范围约束
    
    L_range = ReLU(x - max) + ReLU(min - x)
    
    惩罚超出[min_val, max_val]范围的值
    
    Args:
        x: (B, D) 光谱张量
        min_val: 最小允许值
        max_val: 最大允许值
        
    Returns:
        loss: 标量
    """
    # 超出上界的惩罚
    upper_violation = F.relu(x - max_val)
    # 超出下界的惩罚
    lower_violation = F.relu(min_val - x)
    
    return torch.mean(upper_violation ** 2 + lower_violation ** 2)


def attribute_consistency_loss(x_gen: torch.Tensor, 
                               target_attr: torch.Tensor,
                               predictor: nn.Module) -> torch.Tensor:
    """属性一致性约束
    
    L_attr = ||f(x_gen) - a_cond||²
    
    确保生成光谱与条件属性(香料含量)一致
    
    Args:
        x_gen: (B, D) 生成的光谱
        target_attr: (B, 1) 目标香料含量
        predictor: 预训练的香料含量预测器
        
    Returns:
        loss: 标量
    """
    # 预测器应在eval模式，梯度通过生成器反传
    predicted_attr = predictor(x_gen)
    return F.mse_loss(predicted_attr, target_attr)


class PhysicsConstrainedVAELoss(nn.Module):
    """物理约束cVAE完整损失函数
    
    L = L_recon + β*L_KL + λ₁*L_smooth + λ₂*L_range + λ₃*L_attr
    """
    
    def __init__(self, 
                 beta: float = 0.1,
                 lambda_smooth: float = 1.0,
                 lambda_range: float = 10.0,
                 lambda_attr: float = 5.0,
                 predictor: Optional[nn.Module] = None):
        """
        Args:
            beta: KL散度权重
            lambda_smooth: 平滑约束权重
            lambda_range: 范围约束权重
            lambda_attr: 属性一致性权重
            predictor: 预训练的香料预测器(用于L_attr)
        """
        super().__init__()
        self.beta = beta
        self.lambda_smooth = lambda_smooth
        self.lambda_range = lambda_range
        self.lambda_attr = lambda_attr
        self.predictor = predictor
        
    def forward(self, 
                x: torch.Tensor,
                x_recon: torch.Tensor,
                mu: torch.Tensor,
                logvar: torch.Tensor,
                spice_condition: Optional[torch.Tensor] = None) -> dict:
        """计算完整损失
        
        Args:
            x: (B, D) 原始光谱
            x_recon: (B, D) 重建光谱
            mu: (B, latent_dim) 编码器均值
            logvar: (B, latent_dim) 编码器对数方差
            spice_condition: (B, 1) 条件香料含量 (用于L_attr)
            
        Returns:
            dict: 包含各项损失和总损失
        """
        # 1. 重建损失 (MSE)
        loss_recon = F.mse_loss(x_recon, x)
        
        # 2. KL散度
        # KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 3. 平滑约束
        loss_smooth = smooth_loss(x_recon)
        
        # 4. 范围约束
        loss_range = range_loss(x_recon)
        
        # 5. 属性一致性 (如果有预测器和条件)
        loss_attr = torch.tensor(0.0, device=x.device)
        if self.predictor is not None and spice_condition is not None:
            self.predictor.eval()
            loss_attr = attribute_consistency_loss(
                x_recon, spice_condition, self.predictor
            )
        
        # 总损失
        total_loss = (
            loss_recon + 
            self.beta * loss_kl +
            self.lambda_smooth * loss_smooth +
            self.lambda_range * loss_range +
            self.lambda_attr * loss_attr
        )
        
        return {
            'total': total_loss,
            'recon': loss_recon,
            'kl': loss_kl,
            'smooth': loss_smooth,
            'range': loss_range,
            'attr': loss_attr
        }


class MultiTaskLoss(nn.Module):
    """多任务损失: 分类 + 回归"""
    
    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: 回归损失权重
        """
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self,
                brand_pred: torch.Tensor,
                brand_true: torch.Tensor,
                spice_pred: torch.Tensor,
                spice_true: torch.Tensor) -> dict:
        """
        Args:
            brand_pred: (B, num_classes) 品牌预测logits
            brand_true: (B,) 品牌真实标签
            spice_pred: (B, 1) 香料含量预测
            spice_true: (B, 1) 香料含量真实值
            
        Returns:
            dict: 各项损失
        """
        loss_cls = self.ce_loss(brand_pred, brand_true)
        loss_reg = self.mse_loss(spice_pred, spice_true)
        total = loss_cls + self.alpha * loss_reg
        
        return {
            'total': total,
            'classification': loss_cls,
            'regression': loss_reg
        }
