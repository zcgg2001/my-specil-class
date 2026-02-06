"""
数据增强模块: 基于cVAE的定向光谱增强
"""
import numpy as np
from typing import Dict, Tuple, Optional
try:
    from src.utils.safe_import import torch
except ImportError:
    import torch


class SpectralAugmentor:
    """光谱数据增强器"""
    
    def __init__(self, cvae_model=None, predictor_model=None):
        """
        Args:
            cvae_model: 训练好的cVAE模型
            predictor_model: 训练好的香料含量预测器
        """
        self.cvae = cvae_model
        self.predictor = predictor_model
        
    def get_class_distribution(self, labels: Dict[str, np.ndarray]) -> Dict[Tuple, int]:
        """统计每个类别的样本数
        
        使用 (brand, spice_content) 作为类别key
        """
        class_counts = {}
        brands = labels["brand"]
        spice = labels["spice_content"]
        
        for b, s in zip(brands, spice):
            key = (b, s)
            class_counts[key] = class_counts.get(key, 0) + 1
        
        return class_counts
    
    def identify_minority_classes(self, class_counts: Dict[Tuple, int], 
                                  threshold_ratio: float = 0.5) -> list:
        """识别少数类
        
        Args:
            class_counts: 类别计数
            threshold_ratio: 低于中位数*ratio的类别视为少数类
        """
        counts = list(class_counts.values())
        median = np.median(counts)
        threshold = median * threshold_ratio
        
        minority = [k for k, v in class_counts.items() if v < threshold]
        return minority
    
    def calculate_augment_counts(self, class_counts: Dict[Tuple, int],
                                 max_ratio: float = 3.0) -> Dict[Tuple, int]:
        """计算每个类别需要增强的样本数
        
        Args:
            class_counts: 当前类别计数
            max_ratio: 最大增强倍数
            
        Returns:
            augment_counts: 每个类别需要生成的样本数
        """
        max_count = max(class_counts.values())
        augment_counts = {}
        
        for cls, count in class_counts.items():
            # 目标: 接近最大类但不超过max_ratio倍
            target = min(max_count, int(count * max_ratio))
            augment_counts[cls] = max(0, target - count)
        
        return augment_counts
    
    def add_gaussian_noise(self, spectra: np.ndarray, 
                          sigma: float = 0.01) -> np.ndarray:
        """传统增强: 添加高斯噪声"""
        noise = np.random.normal(0, sigma, spectra.shape)
        return np.clip(spectra + noise, 0, 1)
    
    def add_spectral_shift(self, spectra: np.ndarray,
                          max_shift: int = 2) -> np.ndarray:
        """传统增强: 光谱平移"""
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift > 0:
            return np.pad(spectra[:, :-shift], ((0,0), (shift, 0)), mode='edge')
        elif shift < 0:
            return np.pad(spectra[:, -shift:], ((0,0), (0, -shift)), mode='edge')
        return spectra
    
    def generate_with_cvae(self, brand: int, spice_content: float, 
                          n_samples: int, device: Optional[str] = None) -> np.ndarray:
        """使用cVAE生成光谱
        
        Args:
            brand: 品牌标签
            spice_content: 香料含量
            n_samples: 生成数量
            device: 设备，默认使用cVAE当前设备
            
        Returns:
            generated: (n_samples, 313) 生成的光谱
        """
        if self.cvae is None:
            raise ValueError("cVAE模型未加载")
        
        if device is None:
            device = next(self.cvae.parameters()).device
        else:
            device = torch.device(device)
            self.cvae = self.cvae.to(device)
        
        self.cvae.eval()
        
        # 构建条件向量: [brand_onehot, spice_content_normalized]
        condition = torch.zeros(n_samples, 3, device=device)
        condition[:, brand] = 1.0  # one-hot品牌
        condition[:, 2] = spice_content / 3.6  # 归一化香料含量
        
        # 从先验采样
        latent_dim = getattr(self.cvae, "latent_dim", 64)
        z = torch.randn(n_samples, latent_dim, device=device)
        
        with torch.no_grad():
            generated = self.cvae.decode(z, condition)
        
        return generated.cpu().numpy()
    
    def validate_generated(self, generated: np.ndarray, 
                          target_spice: float,
                          tolerance: float = 0.3) -> np.ndarray:
        """验证生成光谱的属性一致性
        
        使用预测器检查生成光谱的香料含量是否接近目标
        
        Returns:
            valid_samples: 通过验证的样本
        """
        if self.predictor is None:
            return generated
        
        self.predictor.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(generated)
            predicted = self.predictor(x).cpu().numpy().flatten()
        
        # 筛选预测值接近目标的样本
        valid_mask = np.abs(predicted - target_spice) < tolerance
        return generated[valid_mask]
    
    def augment_training_set(self, spectra: np.ndarray, 
                            labels: Dict[str, np.ndarray],
                            method: str = 'cvae',
                            max_ratio: float = 3.0) -> Tuple[np.ndarray, Dict]:
        """增强训练集
        
        Args:
            spectra: 原始光谱
            labels: 标签字典
            method: 'cvae', 'noise', 'shift'
            max_ratio: 最大增强倍数
            
        Returns:
            augmented_spectra: 增强后的光谱
            augmented_labels: 增强后的标签
        """
        class_counts = self.get_class_distribution(labels)
        augment_counts = self.calculate_augment_counts(class_counts, max_ratio)
        
        aug_spectra_list = [spectra]
        aug_brands = [labels["brand"]]
        aug_spice = [labels["spice_content"]]
        aug_batch = [labels["batch"]]
        
        for (brand, spice), n_aug in augment_counts.items():
            if n_aug <= 0:
                continue
            
            if method == 'cvae':
                new_samples = self.generate_with_cvae(brand, spice, n_aug)
            elif method == 'noise':
                # 复制原样本并加噪声
                mask = (labels["brand"] == brand) & (labels["spice_content"] == spice)
                original = spectra[mask]
                idx = np.random.choice(len(original), n_aug, replace=True)
                new_samples = self.add_gaussian_noise(original[idx])
            elif method == 'shift':
                mask = (labels["brand"] == brand) & (labels["spice_content"] == spice)
                original = spectra[mask]
                idx = np.random.choice(len(original), n_aug, replace=True)
                new_samples = self.add_spectral_shift(original[idx])
            else:
                raise ValueError(f"未知增强方法: {method}")
            
            aug_spectra_list.append(new_samples)
            aug_brands.append(np.full(n_aug, brand))
            aug_spice.append(np.full(n_aug, spice))
            aug_batch.append(np.full(n_aug, -1))  # 标记为生成样本
        
        augmented_spectra = np.vstack(aug_spectra_list)
        augmented_labels = {
            "brand": np.concatenate(aug_brands),
            "spice_content": np.concatenate(aug_spice),
            "batch": np.concatenate(aug_batch)
        }
        
        print(f"增强完成: {spectra.shape[0]} -> {augmented_spectra.shape[0]} 样本")
        return augmented_spectra, augmented_labels
