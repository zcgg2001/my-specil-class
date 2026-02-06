"""
评估指标模块

包含:
1. 分类指标: Macro-F1, 混淆矩阵, Per-class Recall
2. 回归指标: RMSE, MAE, R²
3. 生成质量: SAM, MMD, 峰位偏移
"""
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    f1_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)


# =============================================================================
# 分类指标
# =============================================================================

def calculate_classification_metrics(y_true: np.ndarray, 
                                      y_pred: np.ndarray,
                                      class_names: list = None) -> Dict:
    """计算分类评估指标
    
    Args:
        y_true: (N,) 真实标签
        y_pred: (N,) 预测标签
        class_names: 类别名称列表
        
    Returns:
        dict: macro_f1, confusion_matrix, per_class_metrics
    """
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    
    # 每类指标
    report = classification_report(y_true, y_pred, 
                                   target_names=class_names,
                                   output_dict=True)
    
    return {
        'macro_f1': macro_f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'accuracy': np.mean(y_true == y_pred)
    }


# =============================================================================
# 回归指标
# =============================================================================

def calculate_regression_metrics(y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> Dict:
    """计算回归评估指标
    
    Returns:
        dict: rmse, mae, r2
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


# =============================================================================
# 生成质量指标
# =============================================================================

def spectral_angle_mapper(x_real: np.ndarray, 
                          x_gen: np.ndarray) -> float:
    """光谱角匹配度 (SAM)
    
    SAM = arccos(x·y / (||x|| ||y||))
    
    值越小说明光谱越相似 (单位: 度)
    
    Args:
        x_real: (N, D) 真实光谱
        x_gen: (N, D) 生成光谱
        
    Returns:
        mean_sam: 平均SAM (度)
    """
    # 归一化
    x_real_norm = x_real / (np.linalg.norm(x_real, axis=1, keepdims=True) + 1e-8)
    x_gen_norm = x_gen / (np.linalg.norm(x_gen, axis=1, keepdims=True) + 1e-8)
    
    # 计算余弦相似度
    cos_sim = np.sum(x_real_norm * x_gen_norm, axis=1)
    cos_sim = np.clip(cos_sim, -1, 1)  # 数值稳定性
    
    # 转换为角度
    sam_rad = np.arccos(cos_sim)
    sam_deg = np.degrees(sam_rad)
    
    return np.mean(sam_deg)


def gaussian_kernel(x: np.ndarray, y: np.ndarray, 
                    sigma: float = 1.0) -> float:
    """高斯核"""
    diff = x - y
    return np.exp(-np.sum(diff ** 2) / (2 * sigma ** 2))


def maximum_mean_discrepancy(x_real: np.ndarray, 
                             x_gen: np.ndarray,
                             sigma: float = 1.0,
                             n_samples: int = 500) -> float:
    """最大均值差异 (MMD)
    
    衡量两个分布之间的距离
    
    Args:
        x_real: (N1, D) 真实光谱
        x_gen: (N2, D) 生成光谱
        sigma: 高斯核带宽
        n_samples: 采样数量(用于加速计算)
        
    Returns:
        mmd: MMD值
    """
    # 采样以加速
    if len(x_real) > n_samples:
        idx = np.random.choice(len(x_real), n_samples, replace=False)
        x_real = x_real[idx]
    if len(x_gen) > n_samples:
        idx = np.random.choice(len(x_gen), n_samples, replace=False)
        x_gen = x_gen[idx]
    
    n = len(x_real)
    m = len(x_gen)
    
    # K(X, X)
    kxx = 0
    for i in range(n):
        for j in range(i + 1, n):
            kxx += gaussian_kernel(x_real[i], x_real[j], sigma)
    kxx = 2 * kxx / (n * (n - 1)) if n > 1 else 0
    
    # K(Y, Y)
    kyy = 0
    for i in range(m):
        for j in range(i + 1, m):
            kyy += gaussian_kernel(x_gen[i], x_gen[j], sigma)
    kyy = 2 * kyy / (m * (m - 1)) if m > 1 else 0
    
    # K(X, Y)
    kxy = 0
    for i in range(n):
        for j in range(m):
            kxy += gaussian_kernel(x_real[i], x_gen[j], sigma)
    kxy = kxy / (n * m) if n > 0 and m > 0 else 0
    
    mmd = kxx + kyy - 2 * kxy
    return max(0, mmd)  # 确保非负


def find_peaks(spectrum: np.ndarray, 
               threshold: float = 0.5) -> np.ndarray:
    """找到光谱峰位
    
    Args:
        spectrum: (D,) 光谱
        threshold: 相对阈值
        
    Returns:
        peak_indices: 峰位索引
    """
    from scipy.signal import find_peaks as scipy_find_peaks
    
    # 归一化
    spectrum_norm = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + 1e-8)
    
    peaks, _ = scipy_find_peaks(spectrum_norm, height=threshold, distance=5)
    return peaks


def peak_position_error(real_spectra: np.ndarray, 
                        gen_spectra: np.ndarray,
                        wavelengths: np.ndarray = None) -> Dict:
    """峰位偏移统计
    
    Args:
        real_spectra: (N, D) 真实光谱
        gen_spectra: (N, D) 生成光谱
        wavelengths: (D,) 波长数组
        
    Returns:
        dict: mean_shift, std_shift (nm if wavelengths provided)
    """
    if wavelengths is None:
        wavelengths = np.arange(real_spectra.shape[1])
    
    shifts = []
    
    for real, gen in zip(real_spectra, gen_spectra):
        real_peaks = find_peaks(real)
        gen_peaks = find_peaks(gen)
        
        if len(real_peaks) == 0 or len(gen_peaks) == 0:
            continue
        
        # 匹配最近的峰
        for rp in real_peaks:
            distances = np.abs(gen_peaks - rp)
            nearest = gen_peaks[np.argmin(distances)]
            shift = np.abs(wavelengths[nearest] - wavelengths[rp])
            shifts.append(shift)
    
    if len(shifts) == 0:
        return {'mean_shift': 0, 'std_shift': 0, 'n_peaks': 0}
    
    return {
        'mean_shift': np.mean(shifts),
        'std_shift': np.std(shifts),
        'n_peaks': len(shifts)
    }


def calculate_generation_metrics(real_spectra: np.ndarray,
                                 gen_spectra: np.ndarray,
                                 wavelengths: np.ndarray = None) -> Dict:
    """计算完整的生成质量指标
    
    Returns:
        dict: sam, mmd, peak_error
    """
    sam = spectral_angle_mapper(real_spectra, gen_spectra)
    mmd = maximum_mean_discrepancy(real_spectra, gen_spectra)
    peak_err = peak_position_error(real_spectra, gen_spectra, wavelengths)
    
    return {
        'sam_degree': sam,
        'mmd': mmd,
        'peak_mean_shift': peak_err['mean_shift'],
        'peak_std_shift': peak_err['std_shift']
    }


if __name__ == "__main__":
    # 测试分类指标
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    
    cls_metrics = calculate_classification_metrics(y_true, y_pred, ['hongmei', 'yeshu'])
    print(f"Macro-F1: {cls_metrics['macro_f1']:.4f}")
    print(f"Confusion Matrix:\n{cls_metrics['confusion_matrix']}")
    
    # 测试回归指标
    y_true = np.array([2.1, 2.4, 2.7, 3.0])
    y_pred = np.array([2.0, 2.5, 2.6, 3.1])
    
    reg_metrics = calculate_regression_metrics(y_true, y_pred)
    print(f"\nRMSE: {reg_metrics['rmse']:.4f}")
    print(f"R²: {reg_metrics['r2']:.4f}")
    
    # 测试生成指标
    real = np.random.rand(100, 313)
    gen = real + np.random.randn(100, 313) * 0.1
    
    gen_metrics = calculate_generation_metrics(real, gen)
    print(f"\nSAM: {gen_metrics['sam_degree']:.2f}°")
    print(f"MMD: {gen_metrics['mmd']:.6f}")
