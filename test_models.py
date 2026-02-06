"""测试cVAE模型和损失函数"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.utils.safe_import import torch
except ImportError:
    import torch
from src.models.cvae import PhysicsConstrainedCVAE
from src.models.predictor import SpicePredictor
from src.models.multitask import MultiTaskModel
from src.losses.physics_loss import PhysicsConstrainedVAELoss, smooth_loss, range_loss
from src.config import NUM_WAVELENGTHS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# 测试cVAE
print("\n--- 测试 cVAE ---")
cvae = PhysicsConstrainedCVAE(input_dim=NUM_WAVELENGTHS).to(device)
print(f"cVAE参数量: {sum(p.numel() for p in cvae.parameters()):,}")

x = torch.randn(32, NUM_WAVELENGTHS).to(device)
cond = torch.randn(32, 3).to(device)
x_recon, mu, logvar = cvae(x, cond)
print(f"输入形状: {x.shape}, 重建形状: {x_recon.shape}")

# 测试生成
gen = cvae.generate(cond[:1], n_samples=5)
print(f"生成形状: {gen.shape}")

# 测试损失函数
print("\n--- 测试损失函数 ---")
l_smooth = smooth_loss(x_recon)
l_range = range_loss(x_recon)
print(f"L_smooth: {l_smooth.item():.4f}")
print(f"L_range: {l_range.item():.4f}")

criterion = PhysicsConstrainedVAELoss()
losses = criterion(x, x_recon, mu, logvar)
print(f"总损失: {losses['total'].item():.4f}")

# 测试预测器
print("\n--- 测试 SpicePredictor ---")
predictor = SpicePredictor().to(device)
print(f"预测器参数量: {sum(p.numel() for p in predictor.parameters()):,}")
pred = predictor(x)
print(f"预测形状: {pred.shape}")

# 测试多任务模型
print("\n--- 测试 MultiTaskModel ---")
mtl = MultiTaskModel().to(device)
print(f"多任务模型参数量: {sum(p.numel() for p in mtl.parameters()):,}")
cls, reg = mtl(x)
print(f"分类输出: {cls.shape}, 回归输出: {reg.shape}")

print("\n所有模型测试通过!")
