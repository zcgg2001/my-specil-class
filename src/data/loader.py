"""
数据加载模块: 从CSV文件加载近红外光谱数据
"""
import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import interp1d
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, BRANDS, SPICE_LEVELS, BATCHES, 
                    WAVELENGTH_START, WAVELENGTH_END, WAVELENGTH_STEP, NUM_WAVELENGTHS)


class NIRDataLoader:
    """近红外光谱数据加载器"""
    
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        # 目标波长网格 (200-1760nm, 步长5nm)
        self.target_wavelengths = np.linspace(
            WAVELENGTH_START, WAVELENGTH_END, NUM_WAVELENGTHS
        )
        
    def load_single_csv(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载单个CSV文件
        
        Returns:
            wavelengths: 波长数组
            counts: 光谱计数值
        """
        # 尝试不同编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                wavelengths = df.iloc[:, 0].values
                counts = df.iloc[:, 1].values
                return wavelengths, counts
            except UnicodeDecodeError:
                continue
            except Exception as e:
                # 其他错误直接抛出
                raise e
        raise ValueError(f"无法解码文件: {filepath}")
    
    def resample_spectrum(self, wavelengths: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """将光谱重采样到目标波长网格
        
        Args:
            wavelengths: 原始波长
            counts: 原始计数值
            
        Returns:
            resampled: 重采样后的光谱 (313维)
        """
        # 创建插值函数
        f = interp1d(wavelengths, counts, kind='linear', 
                     bounds_error=False, fill_value='extrapolate')
        resampled = f(self.target_wavelengths)
        return resampled
    
    def parse_filepath(self, filepath: str) -> Dict:
        """从文件路径解析元数据
        
        Returns:
            dict with keys: brand, spice_content, batch
        """
        parts = filepath.replace("\\", "/").split("/")
        
        # 解析品牌
        brand_folder = None
        for part in parts:
            if part in BRANDS:
                brand_folder = part
                break
        brand_idx = BRANDS.index(brand_folder) if brand_folder else 0
        
        # 解析香料含量
        spice_content = 0.0
        for part in parts:
            for level in SPICE_LEVELS:
                if f"{level}%" in part or f"{level:.1f}%" in part:
                    spice_content = level
                    break
        
        # 解析批次
        batch = "01"
        for part in parts:
            if part in BATCHES:
                batch = part
                break
        batch_idx = BATCHES.index(batch)
        
        return {
            "brand": brand_idx,
            "brand_name": brand_folder,
            "spice_content": spice_content,
            "batch": batch_idx,
            "batch_name": batch
        }
    
    def load_all_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """加载所有数据
        
        Returns:
            spectra: (N, 313) 光谱矩阵
            labels: dict containing:
                - brand: (N,) 品牌标签
                - spice_content: (N,) 香料含量
                - batch: (N,) 批次标签
        """
        all_spectra = []
        all_brands = []
        all_spice = []
        all_batches = []
        all_ids = []
        
        # 遍历所有CSV文件
        csv_pattern = os.path.join(self.data_dir, "**", "*.csv")
        csv_files = glob.glob(csv_pattern, recursive=True)
        
        print(f"发现 {len(csv_files)} 个CSV文件")
        
        for i, filepath in enumerate(csv_files):
            try:
                # 加载并重采样
                wavelengths, counts = self.load_single_csv(filepath)
                spectrum = self.resample_spectrum(wavelengths, counts)
                
                # 解析元数据
                meta = self.parse_filepath(filepath)
                
                all_spectra.append(spectrum)
                all_brands.append(meta["brand"])
                all_spice.append(meta["spice_content"])
                all_batches.append(meta["batch"])
                all_ids.append(os.path.basename(filepath))
                
                if (i + 1) % 500 == 0:
                    print(f"已加载 {i + 1}/{len(csv_files)} 个文件")
                    
            except Exception as e:
                print(f"加载失败: {filepath}, 错误: {e}")
                continue
        
        spectra = np.array(all_spectra)
        labels = {
            "brand": np.array(all_brands),
            "spice_content": np.array(all_spice),
            "batch": np.array(all_batches),
            "sample_id": np.array(all_ids)
        }
        
        print(f"数据加载完成: {spectra.shape[0]} 样本, {spectra.shape[1]} 维")
        return spectra, labels


def load_all_data(data_dir: str = DATA_DIR) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """便捷函数: 加载所有数据"""
    loader = NIRDataLoader(data_dir)
    return loader.load_all_data()


if __name__ == "__main__":
    # 测试数据加载
    spectra, labels = load_all_data()
    print(f"光谱形状: {spectra.shape}")
    print(f"品牌分布: {np.bincount(labels['brand'])}")
    print(f"香料含量: {np.unique(labels['spice_content'])}")
    print(f"批次分布: {np.bincount(labels['batch'])}")
