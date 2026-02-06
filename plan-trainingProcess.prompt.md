## 完整训练流程

> **注意：此项目要求在 GPU 上运行，请确保 CUDA 可用。**

### 0. 前置检查：确保 GPU 可用

```bash
# 检查 CUDA 是否可用
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# 如果显示 CUDA: False，需要安装 GPU 版本的 PyTorch
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 测试数据加载是否正常
```bash
python test_loader.py
```

### 3. 测试模型是否正常
```bash
python test_models.py
```

### 4. 分步训练（推荐）

**Step 1: 训练香料含量预测器（用于后续 cVAE 的 L_attr 约束）**
```bash
python train_predictor.py --epochs 50
```

**Step 2: 训练物理约束 cVAE（用于数据增强）**
```bash
python train_cvae.py --epochs 100 --model physics
```

**Step 3: 训练多任务模型（分类+回归）**

随机划分 + 无增强：
```bash
python train_multitask.py --epochs 100 --split random --augment none
```

随机划分 + 物理约束 cVAE 增强：
```bash
python train_multitask.py --epochs 100 --split random --augment cvae_physics
```

批次划分（域外测试）+ 无增强：
```bash
python train_multitask.py --epochs 100 --split batch --augment none
```

批次划分 + 物理约束 cVAE 增强：
```bash
python train_multitask.py --epochs 100 --split batch --augment cvae_physics
```

---

### 5. 一键运行所有实验（可选）

如果您想运行完整的实验矩阵（E1-E7），可以使用：

```bash
python run_experiments.py
```

或者指定特定实验：
```bash
# 只运行 E1 和 E4
python run_experiments.py --exp-ids E1,E4

# 跳过预训练步骤（如果已训练过 predictor 和 cVAE）
python run_experiments.py --skip-pretrain
```

---

### 6. 实验配置说明

| 实验 | 划分方式 | 增强方法 | 命令 |
|------|----------|----------|------|
| E1 | random | none | `python train_multitask.py --split random --augment none` |
| E2 | random | noise | `python train_multitask.py --split random --augment noise` |
| E3 | random | vae_no_physics | `python train_multitask.py --split random --augment vae_no_physics` |
| E4 | random | cvae_physics | `python train_multitask.py --split random --augment cvae_physics` |
| E5 | batch | none | `python train_multitask.py --split batch --augment none` |
| E6 | batch | cvae_physics | `python train_multitask.py --split batch --augment cvae_physics` |
| E7 | batch (留一交叉验证) | cvae_physics | 由 `run_experiments.py` 自动执行 |

---

### 7. 训练结果位置

- **模型权重**: `checkpoints/` 目录
- **实验结果**: `results/experiment_summary.json`

