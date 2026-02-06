"""
Step 1: 训练香料含量预测器

用于后续cVAE的L_attr约束
"""
import os
import sys
import argparse
import pickle
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
from src.models.predictor import SpicePredictor, SimpleMLP
from src.evaluation.metrics import calculate_regression_metrics
from src.utils.split_utils import resolve_split


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            pred = model(x)
            all_pred.append(pred.cpu().numpy())
            all_true.append(y.numpy())

    y_pred = np.concatenate(all_pred).flatten()
    y_true = np.concatenate(all_true).flatten()
    return calculate_regression_metrics(y_true, y_pred)


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
    X_train_np = preprocessor.preprocess_pipeline(spectra[split['train']], fit=True, use_smooth=True)
    X_val_np = preprocessor.preprocess_pipeline(spectra[split['val']], fit=False, use_smooth=True)
    X_test_np = preprocessor.preprocess_pipeline(spectra[split['test']], fit=False, use_smooth=True)

    X_train = torch.FloatTensor(X_train_np)
    y_train = torch.FloatTensor(labels['spice_content'][split['train']]).unsqueeze(1)
    X_val = torch.FloatTensor(X_val_np)
    y_val = torch.FloatTensor(labels['spice_content'][split['val']]).unsqueeze(1)
    X_test = torch.FloatTensor(X_test_np)
    y_test = torch.FloatTensor(labels['spice_content'][split['test']]).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

    model = SpicePredictor().to(device) if args.model == 'cnn' else SimpleMLP().to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"spice_predictor_best_{split_tag}.pth")

    best_val_rmse = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics['rmse'])

        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            torch.save(model.state_dict(), model_path)

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val RMSE: {val_metrics['rmse']:.4f} | "
                f"Val R2: {val_metrics['r2']:.4f}"
            )

    model.load_state_dict(torch.load(model_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    print("\n" + "=" * 50)
    print("测试集结果:")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    print("=" * 50)

    preprocessor_path = os.path.join(MODEL_DIR, f"preprocessor_{split_tag}.pkl")
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)

    # 兼容旧流程：random 默认名也保存
    if split_tag == "random":
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'spice_predictor_best.pth'))
        with open(os.path.join(MODEL_DIR, 'preprocessor.pkl'), 'wb') as f:
            pickle.dump(preprocessor, f)

    print(f"\n模型已保存到: {model_path}")
    print(f"预处理器已保存到: {preprocessor_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练香料含量预测器")
    parser.add_argument('--epochs', type=int, default=EPOCHS_PREDICTOR)
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'mlp'])
    parser.add_argument('--split', type=str, default='random', choices=['random', 'batch'])
    parser.add_argument('--test-batch', type=int, default=None,
                        help="当 split=batch 时，指定测试批次ID (0-3)")
    args = parser.parse_args()
    main(args)
