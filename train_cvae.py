"""
Step 2: 训练物理约束cVAE

用于光谱数据增强
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Safe PyTorch import to handle Intel JIT compiler conflicts
try:
    from src.utils.safe_import import torch
except ImportError:
    import torch

from torch.utils.data import DataLoader, TensorDataset
from src.config import *
from src.data.loader import load_all_data
from src.data.preprocessor import NIRPreprocessor
from src.models.cvae import PhysicsConstrainedCVAE
from src.models.predictor import SpicePredictor
from src.losses.physics_loss import PhysicsConstrainedVAELoss
from src.evaluation.metrics import calculate_generation_metrics
from src.utils.split_utils import resolve_split


def prepare_condition(brands, spice_contents, device):
    """准备条件向量: [brand_onehot(2), spice_normalized(1)]"""
    batch_size = len(brands)
    condition = torch.zeros(batch_size, 3, device=device)

    for i, b in enumerate(brands):
        condition[i, int(b)] = 1.0

    condition[:, 2] = torch.tensor(spice_contents, device=device, dtype=torch.float32) / 3.6
    return condition


def train_epoch(model, dataloader, criterion, optimizer, device, brands, spices):
    model.train()
    total_losses = {'total': 0.0, 'recon': 0.0, 'kl': 0.0, 'smooth': 0.0, 'range': 0.0, 'attr': 0.0}

    offset = 0
    for (x,) in dataloader:
        x = x.to(device)
        batch_size = x.size(0)

        brands_batch = brands[offset:offset + batch_size]
        spice_batch = spices[offset:offset + batch_size]
        offset += batch_size

        condition = prepare_condition(brands_batch, spice_batch, device)
        spice_tensor = torch.tensor(spice_batch, device=device, dtype=torch.float32).unsqueeze(1) / 3.6

        optimizer.zero_grad()
        x_recon, mu, logvar = model(x, condition)
        losses = criterion(x, x_recon, mu, logvar, spice_tensor)
        losses['total'].backward()
        optimizer.step()

        for k, v in losses.items():
            total_losses[k] += v.item()

    n_batches = len(dataloader)
    return {k: v / n_batches for k, v in total_losses.items()}


def evaluate_generation(model, dataloader, brands, spices, device):
    """评估生成质量"""
    model.eval()
    all_real, all_gen = [], []

    offset = 0
    with torch.no_grad():
        for (x,) in dataloader:
            x = x.to(device)
            batch_size = x.size(0)

            brands_batch = brands[offset:offset + batch_size]
            spice_batch = spices[offset:offset + batch_size]
            offset += batch_size

            condition = prepare_condition(brands_batch, spice_batch, device)
            generated = model.generate(condition, n_samples=1)

            all_real.append(x.cpu().numpy())
            all_gen.append(generated.cpu().numpy())

    real = np.concatenate(all_real)
    gen = np.concatenate(all_gen)
    return calculate_generation_metrics(real, gen)


def main(args):
    if args.epochs <= 0:
        raise ValueError("--epochs 必须大于 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"使用设备: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"使用设备: {device}")

    print("加载数据...")
    spectra, labels = load_all_data()

    split, split_tag, train_batches, test_batches = resolve_split(
        labels=labels,
        n_samples=len(spectra),
        split_mode=args.split,
        test_batch=args.test_batch,
        random_seed=RANDOM_SEED,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
    )

    if train_batches is not None:
        print(f"批次划分: train={train_batches}, test={test_batches}")
    print(f"划分方式: {args.split}")
    print(f"训练/验证/测试: {len(split['train'])}/{len(split['val'])}/{len(split['test'])}")
    print(f"Split标签: {split_tag}")

    # 只在训练集拟合预处理器，避免数据泄漏
    preprocessor = NIRPreprocessor()
    X_train_np = preprocessor.preprocess_pipeline(spectra[split['train']], fit=True)
    X_val_np = preprocessor.preprocess_pipeline(spectra[split['val']], fit=False)

    train_brands = labels['brand'][split['train']]
    train_spice = labels['spice_content'][split['train']]
    val_brands = labels['brand'][split['val']]
    val_spice = labels['spice_content'][split['val']]

    X_train = torch.FloatTensor(X_train_np)
    train_loader = DataLoader(TensorDataset(X_train), batch_size=BATCH_SIZE, shuffle=False)

    X_val = torch.FloatTensor(X_val_np)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=BATCH_SIZE, shuffle=False)

    predictor = None
    predictor_path = os.path.join(MODEL_DIR, f"spice_predictor_best_{split_tag}.pth")
    if args.use_attr_loss:
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(
                f"未找到与当前split匹配的预测器: {predictor_path}\n"
                "请先运行 train_predictor.py 并使用相同的 --split/--test-batch 参数。"
            )
        print("加载预训练香料预测器...")
        predictor = SpicePredictor().to(device)
        predictor.load_state_dict(torch.load(predictor_path, map_location=device))
        predictor.eval()
        for p in predictor.parameters():
            p.requires_grad = False

    model = PhysicsConstrainedCVAE(
        input_dim=NUM_WAVELENGTHS,
        condition_dim=3,
        hidden_dims=HIDDEN_DIMS,
        latent_dim=LATENT_DIM,
    ).to(device)
    print(f"cVAE参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = PhysicsConstrainedVAELoss(
        beta=BETA_KL,
        lambda_smooth=LAMBDA_SMOOTH if args.use_physics else 0,
        lambda_range=LAMBDA_RANGE if args.use_physics else 0,
        lambda_attr=LAMBDA_ATTR if args.use_attr_loss and predictor is not None else 0,
        predictor=predictor,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(MODEL_DIR, exist_ok=True)
    variant = 'physics' if args.use_physics else 'vanilla'
    save_name = f"cvae_{variant}_{split_tag}.pth"
    save_path = os.path.join(MODEL_DIR, save_name)

    best_sam = float('inf')
    for epoch in range(args.epochs):
        losses = train_epoch(model, train_loader, criterion, optimizer, device, train_brands, train_spice)

        should_eval = ((epoch + 1) % 20 == 0) or (epoch == args.epochs - 1)
        if should_eval:
            val_metrics = evaluate_generation(model, val_loader, val_brands, val_spice, device)
            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Loss: {losses['total']:.4f} (R:{losses['recon']:.4f} "
                f"KL:{losses['kl']:.4f} S:{losses['smooth']:.4f} "
                f"Rg:{losses['range']:.4f} A:{losses['attr']:.4f}) | "
                f"SAM: {val_metrics['sam_degree']:.2f}°"
            )

            if val_metrics['sam_degree'] < best_sam:
                best_sam = val_metrics['sam_degree']
                torch.save(model.state_dict(), save_path)

    if best_sam == float('inf'):
        torch.save(model.state_dict(), save_path)
        print("\n未执行验证，已保存最后一轮模型。")
    else:
        print(f"\n最佳SAM: {best_sam:.2f}°")

    # 兼容旧流程：random 默认名也保存
    if split_tag == "random":
        legacy_name = 'cvae_physics.pth' if args.use_physics else 'cvae_vanilla.pth'
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, legacy_name))

    print(f"模型已保存到: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练物理约束cVAE")
    parser.add_argument('--epochs', type=int, default=EPOCHS_CVAE)
    parser.add_argument('--split', type=str, default='random', choices=['random', 'batch'])
    parser.add_argument('--test-batch', type=int, default=None,
                        help="当 split=batch 时，指定测试批次ID (0-3)")
    parser.add_argument('--use-physics', action='store_true', default=True, help="使用物理约束(平滑+范围)")
    parser.add_argument('--no-physics', dest='use_physics', action='store_false')
    parser.add_argument('--use-attr-loss', action='store_true', default=True, help="使用属性一致性约束")
    parser.add_argument('--no-attr-loss', dest='use_attr_loss', action='store_false')
    args = parser.parse_args()
    main(args)
