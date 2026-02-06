"""
数据预处理模块
"""
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler


class NIRPreprocessor:
    """近红外光谱预处理器"""
    
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
        
    def handle_missing(self, spectra: np.ndarray) -> np.ndarray:
        """处理缺失值: 使用线性插值
        
        Args:
            spectra: (N, D) 光谱矩阵
            
        Returns:
            filled: 填充后的光谱
        """
        filled = spectra.copy()
        for i in range(filled.shape[0]):
            mask = np.isnan(filled[i]) | np.isinf(filled[i])
            if mask.any():
                x = np.arange(len(filled[i]))
                filled[i][mask] = np.interp(
                    x[mask], x[~mask], filled[i][~mask]
                )
        return filled
    
    def detect_outliers(self, spectra: np.ndarray, sigma: float = 3.0) -> np.ndarray:
        """检测异常值 (3σ原则)
        
        Returns:
            mask: (N,) 布尔数组，True表示异常样本
        """
        mean = np.mean(spectra, axis=0)
        std = np.std(spectra, axis=0)
        
        # 计算每个样本的异常波段数
        lower = mean - sigma * std
        upper = mean + sigma * std
        
        outlier_bands = (spectra < lower) | (spectra > upper)
        outlier_ratio = outlier_bands.sum(axis=1) / spectra.shape[1]
        
        # 如果超过10%的波段异常，标记为异常样本
        return outlier_ratio > 0.1
    
    def normalize(self, spectra: np.ndarray, fit: bool = True) -> np.ndarray:
        """MinMax归一化到[0, 1]
        
        Args:
            spectra: (N, D) 光谱矩阵
            fit: 是否拟合scaler
            
        Returns:
            normalized: 归一化后的光谱
        """
        if fit:
            self.scaler.fit(spectra)
            self.is_fitted = True
        return self.scaler.transform(spectra)
    
    def inverse_normalize(self, spectra: np.ndarray) -> np.ndarray:
        """逆归一化"""
        if not self.is_fitted:
            raise ValueError("Scaler未拟合，请先调用normalize(fit=True)")
        return self.scaler.inverse_transform(spectra)
    
    def snv_transform(self, spectra: np.ndarray) -> np.ndarray:
        """标准正态变量变换 (SNV)
        
        对每个样本进行标准化
        """
        mean = spectra.mean(axis=1, keepdims=True)
        std = spectra.std(axis=1, keepdims=True)
        return (spectra - mean) / (std + 1e-8)
    
    def msc_transform(self, spectra: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
        """多元散射校正 (MSC)
        
        Args:
            spectra: (N, D) 光谱矩阵
            reference: 参考光谱，默认使用均值
        """
        if reference is None:
            reference = spectra.mean(axis=0)
        
        corrected = np.zeros_like(spectra)
        for i in range(spectra.shape[0]):
            # 线性回归: spectrum = a * reference + b
            coeffs = np.polyfit(reference, spectra[i], 1)
            corrected[i] = (spectra[i] - coeffs[1]) / coeffs[0]
        
        return corrected
    
    def savgol_smooth(self, spectra: np.ndarray, 
                      window_length: int = 11, polyorder: int = 2) -> np.ndarray:
        """Savitzky-Golay平滑
        
        Args:
            spectra: (N, D) 光谱矩阵
            window_length: 窗口长度(奇数)
            polyorder: 多项式阶数
        """
        from scipy.signal import savgol_filter
        return savgol_filter(spectra, window_length, polyorder, axis=1)
    
    def first_derivative(self, spectra: np.ndarray) -> np.ndarray:
        """一阶导数"""
        return np.gradient(spectra, axis=1)
    
    def second_derivative(self, spectra: np.ndarray) -> np.ndarray:
        """二阶导数"""
        return np.gradient(np.gradient(spectra, axis=1), axis=1)
    
    def preprocess_pipeline(self, spectra: np.ndarray, 
                           fit: bool = True,
                           use_snv: bool = False,
                           use_msc: bool = False,
                           use_smooth: bool = True) -> np.ndarray:
        """完整预处理流程
        
        1. 缺失值处理
        2. SNV/MSC (可选)
        3. 平滑 (可选)
        4. MinMax归一化
        """
        # 1. 缺失值
        processed = self.handle_missing(spectra)
        
        # 2. 散射校正
        if use_snv:
            processed = self.snv_transform(processed)
        elif use_msc:
            processed = self.msc_transform(processed)
        
        # 3. 平滑
        if use_smooth:
            processed = self.savgol_smooth(processed)
        
        # 4. 归一化
        processed = self.normalize(processed, fit=fit)
        
        return processed


if __name__ == "__main__":
    # 测试预处理
    from loader import load_all_data
    spectra, labels = load_all_data()
    
    preprocessor = NIRPreprocessor()
    processed = preprocessor.preprocess_pipeline(spectra)
    
    print(f"原始范围: [{spectra.min():.2f}, {spectra.max():.2f}]")
    print(f"处理后范围: [{processed.min():.4f}, {processed.max():.4f}]")
