"""
Step 3: 训练多任务模型 (分类+回归)

支持数据增强对比实验
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Safe PyTorch import to handle Intel JIT compiler conflicts
try:
    from src.utils.safe_import import torch, nn
except ImportError:
    import torch
    import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from src.config import *
from src.data.loader import load_all_data
from src.data.preprocessor import NIRPreprocessor
from src.data.augmentor import SpectralAugmentor
from src.models.multitask import MultiTaskModel
from src.models.cvae import PhysicsConstrainedCVAE
from src.losses.physics_loss import MultiTaskLoss
from src.evaluation.metrics import calculate_classification_metrics, calculate_regression_metrics
from src.utils.split_utils import resolve_split


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for x, brand, spice in dataloader:
        x = x.to(device)
        brand = brand.to(device)
        spice = spice.to(device)

        optimizer.zero_grad()
        cls_logits, reg_pred = model(x)
        losses = criterion(cls_logits, brand, reg_pred, spice)
        losses['total'].backward()
        optimizer.step()
        total_loss += losses['total'].item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_cls_pred, all_cls_true = [], []
    all_reg_pred, all_reg_true = [], []

    with torch.no_grad():
        for x, brand, spice in dataloader:
            x = x.to(device)
            cls_logits, reg_pred = model(x)

            cls_pred = cls_logits.argmax(dim=1).cpu().numpy()
            all_cls_pred.append(cls_pred)
            all_cls_true.append(brand.numpy())
            all_reg_pred.append(reg_pred.cpu().numpy())
            all_reg_true.append(spice.numpy())

    cls_pred = np.concatenate(all_cls_pred)
    cls_true = np.concatenate(all_cls_true)
    reg_pred = np.concatenate(all_reg_pred).flatten()
    reg_true = np.concatenate(all_reg_true).flatten()

    cls_metrics = calculate_classification_metrics(cls_true, cls_pred, BRANDS)
    reg_metrics = calculate_regression_metrics(reg_true, reg_pred)
    return {**cls_metrics, **reg_metrics}


def apply_augmentation(x_train, labels_train, augment_method, device, split_tag):
    """应用数据增强（仅基于训练集）。"""
    if augment_method == 'none':
        return x_train, labels_train

    augmentor = SpectralAugmentor()

    if augment_method == 'noise':
        x_aug, labels_aug = augmentor.augment_training_set(
            x_train,
            labels_train,
            method='noise',
            max_ratio=AUGMENT_RATIO_MAX,
        )
        return x_aug, labels_aug

    if augment_method in ['cvae_physics', 'vae_no_physics']:
        cvae = PhysicsConstrainedCVAE().to(device)
        variant = 'physics' if augment_method == 'cvae_physics' else 'vanilla'
        cvae_path = os.path.join(MODEL_DIR, f"cvae_{variant}_{split_tag}.pth")

        if not os.path.exists(cvae_path):
            raise FileNotFoundError(
                f"未找到与当前split匹配的cVAE模型: {cvae_path}\n"
                "请先运行 train_cvae.py 并使用相同的 --split/--test-batch 参数。"
            )

        cvae.load_state_dict(torch.load(cvae_path, map_location=device))
        cvae.eval()
        augmentor.cvae = cvae

        x_aug, labels_aug = augmentor.augment_training_set(
            x_train,
            labels_train,
            method='cvae',
            max_ratio=AUGMENT_RATIO_MAX,
        )
        return x_aug, labels_aug

    raise ValueError(f"未知增强方法: {augment_method}")


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
    x_train_base = preprocessor.preprocess_pipeline(spectra[split['train']], fit=True)
    x_val = preprocessor.preprocess_pipeline(spectra[split['val']], fit=False)
    x_test = preprocessor.preprocess_pipeline(spectra[split['test']], fit=False)

    labels_train_base = {
        'brand': labels['brand'][split['train']],
        'spice_content': labels['spice_content'][split['train']],
        'batch': labels['batch'][split['train']],
    }

    print(f"增强方法: {args.augment}")
    x_train, labels_train = apply_augmentation(
        x_train=x_train_base,
        labels_train=labels_train_base,
        augment_method=args.augment,
        device=device,
        split_tag=split_tag,
    )

    train_dataset = TensorDataset(
        torch.FloatTensor(x_train),
        torch.LongTensor(labels_train['brand']),
        torch.FloatTensor(labels_train['spice_content']).unsqueeze(1),
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TensorDataset(
        torch.FloatTensor(x_val),
        torch.LongTensor(labels['brand'][split['val']]),
        torch.FloatTensor(labels['spice_content'][split['val']]).unsqueeze(1),
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    test_dataset = TensorDataset(
        torch.FloatTensor(x_test),
        torch.LongTensor(labels['brand'][split['test']]),
        torch.FloatTensor(labels['spice_content'][split['test']]).unsqueeze(1),
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = MultiTaskModel().to(device)
    criterion = MultiTaskLoss(alpha=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_name = f"multitask_{args.split}_{args.augment}_{split_tag}.pth"
    model_path = os.path.join(MODEL_DIR, model_name)

    best_f1 = 0.0
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            val_metrics = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Val F1: {val_metrics['macro_f1']:.4f} | "
                f"Val RMSE: {val_metrics['rmse']:.4f}"
            )

            if val_metrics['macro_f1'] > best_f1 or epoch == args.epochs - 1:
                best_f1 = val_metrics['macro_f1']
                torch.save(model.state_dict(), model_path)

    model.load_state_dict(torch.load(model_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    print("\n" + "=" * 60)
    print(f"实验配置: split={args.split}, augment={args.augment}, split_tag={split_tag}")
    print("=" * 60)
    print("分类结果:")
    print(f"  Macro-F1:  {test_metrics['macro_f1']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print("\n回归结果:")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  R2:   {test_metrics['r2']:.4f}")
    print("\n混淆矩阵:")
    print(test_metrics['confusion_matrix'])
    print("=" * 60)

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练多任务模型")
    parser.add_argument('--epochs', type=int, default=EPOCHS_MULTITASK)
    parser.add_argument('--split', type=str, default='random', choices=['random', 'batch'])
    parser.add_argument('--test-batch', type=int, default=None,
                        help="当 split=batch 时，指定测试批次ID (0-3)")
    parser.add_argument('--augment', type=str, default='none',
                        choices=['none', 'noise', 'vae_no_physics', 'cvae_physics'])
    args = parser.parse_args()
    main(args)
