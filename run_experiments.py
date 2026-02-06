"""
运行完整实验矩阵

E1-E4: 随机划分 + 不同增强方法
E5-E6: 批次划分(域外测试) + 增强对比
E7: 留一批次交叉验证
"""
import os
import sys
import argparse
import json
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import RESULT_DIR, MODEL_DIR
from src.utils.split_utils import build_split_tag


def run_cmd(cmd):
    """Run command and stream output after completion."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    return result.returncode == 0


def ensure_predictor(split, test_batch, epochs, force_retrain=False):
    """Ensure split-specific spice predictor checkpoint exists."""
    split_tag = build_split_tag(split, test_batch)
    model_path = os.path.join(MODEL_DIR, f"spice_predictor_best_{split_tag}.pth")
    if os.path.exists(model_path) and not force_retrain:
        print(f"复用预测器: {model_path}")
        return True

    print("\n" + "=" * 60)
    print(f"预训练预测器: split={split}, test_batch={test_batch}")
    print("=" * 60)

    cmd = [
        sys.executable,
        "train_predictor.py",
        "--epochs",
        str(epochs),
        "--split",
        split,
    ]
    if split == "batch" and test_batch is not None:
        cmd.extend(["--test-batch", str(test_batch)])

    return run_cmd(cmd)


def ensure_cvae(split, test_batch, variant, epochs, force_retrain=False):
    """Ensure split-specific cVAE checkpoint exists."""
    split_tag = build_split_tag(split, test_batch)
    model_path = os.path.join(MODEL_DIR, f"cvae_{variant}_{split_tag}.pth")
    if os.path.exists(model_path) and not force_retrain:
        print(f"复用cVAE: {model_path}")
        return True

    print("\n" + "=" * 60)
    print(f"预训练cVAE: variant={variant}, split={split}, test_batch={test_batch}")
    print("=" * 60)

    cmd = [
        sys.executable,
        "train_cvae.py",
        "--epochs",
        str(epochs),
        "--split",
        split,
    ]
    if split == "batch" and test_batch is not None:
        cmd.extend(["--test-batch", str(test_batch)])

    if variant == "physics":
        cmd.extend(["--use-physics", "--use-attr-loss"])
    elif variant == "vanilla":
        cmd.extend(["--no-physics", "--no-attr-loss"])
    else:
        raise ValueError(f"未知cVAE类型: {variant}")

    return run_cmd(cmd)


def ensure_pretrain_for_experiment(split, augment, test_batch, args):
    """Prepare only the pretraining assets needed by current experiment."""
    if augment == "cvae_physics":
        ok = ensure_predictor(split, test_batch, args.predictor_epochs, args.force_pretrain)
        if not ok:
            return False
        return ensure_cvae(split, test_batch, "physics", args.cvae_epochs, args.force_pretrain)

    if augment == "vae_no_physics":
        return ensure_cvae(split, test_batch, "vanilla", args.cvae_epochs, args.force_pretrain)

    return True


def run_experiment(exp_id, split, augment, epochs=100, test_batch=None):
    """运行单个实验。"""
    cmd = [
        sys.executable,
        "train_multitask.py",
        "--split",
        split,
        "--augment",
        augment,
        "--epochs",
        str(epochs),
    ]
    if test_batch is not None:
        cmd.extend(["--test-batch", str(test_batch)])

    print("\n" + "=" * 60)
    print(f"运行实验 {exp_id}: split={split}, augment={augment}, test_batch={test_batch}")
    print("=" * 60)
    return run_cmd(cmd)


def run_batch_cv(epochs, args):
    """E7: 留一批次交叉验证。"""
    cv_results = {}

    for test_batch in range(4):
        print("\n" + "=" * 60)
        print(f"E7 Fold {test_batch+1}/4: 测试批次={test_batch}")
        print("=" * 60)

        if not args.skip_pretrain:
            ok = ensure_pretrain_for_experiment("batch", "cvae_physics", test_batch, args)
            if not ok:
                cv_results[f"fold_{test_batch}"] = False
                continue

        success = run_experiment(
            exp_id=f"E7_fold{test_batch}",
            split="batch",
            augment="cvae_physics",
            epochs=epochs,
            test_batch=test_batch,
        )
        cv_results[f"fold_{test_batch}"] = success

    return {
        "success": all(cv_results.values()) if cv_results else False,
        "folds": cv_results,
    }


def main(args):
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU 检测: {torch.cuda.get_device_name(0)}")
        else:
            print("警告: CUDA 不可用，将使用CPU运行。")
    except ImportError:
        print("警告: 无法导入 torch 检查 GPU")

    os.makedirs(RESULT_DIR, exist_ok=True)

    experiments = {
        "E1": ("random", "none"),
        "E2": ("random", "noise"),
        "E3": ("random", "vae_no_physics"),
        "E4": ("random", "cvae_physics"),
        "E5": ("batch", "none"),
        "E6": ("batch", "cvae_physics"),
    }

    if args.exp_ids:
        exp_list = [exp.strip() for exp in args.exp_ids.split(",") if exp.strip()]
    else:
        exp_list = list(experiments.keys())

    results = {}

    for exp_id in exp_list:
        if exp_id == "E7":
            details = run_batch_cv(args.epochs, args)
            results[exp_id] = details
            continue

        if exp_id not in experiments:
            print(f"未知实验ID: {exp_id}")
            continue

        split, augment = experiments[exp_id]
        test_batch = 3 if split == "batch" else None

        if not args.skip_pretrain:
            ok = ensure_pretrain_for_experiment(split, augment, test_batch, args)
            if not ok:
                results[exp_id] = {"success": False, "reason": "pretrain_failed"}
                continue

        success = run_experiment(exp_id, split, augment, args.epochs, test_batch)
        results[exp_id] = {"success": success}

    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiments": results,
    }

    summary_path = os.path.join(RESULT_DIR, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("所有实验完成!")
    print(f"结果保存在: {RESULT_DIR}")
    print(f"摘要文件: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行完整实验")
    parser.add_argument("--exp-ids", type=str, default=None, help="要运行的实验ID，逗号分隔，如 E1,E2,E4")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--skip-pretrain", action="store_true", help="跳过预训练步骤")
    parser.add_argument("--predictor-epochs", type=int, default=50, help="预训练预测器轮数")
    parser.add_argument("--cvae-epochs", type=int, default=100, help="预训练cVAE轮数")
    parser.add_argument("--force-pretrain", action="store_true", help="强制重训并覆盖已有预训练模型")
    args = parser.parse_args()
    main(args)
