"""
域划分策略: 用于域迁移/泛化实验
"""
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split


class DomainSplitter:
    """域划分器: 支持多种划分策略"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def random_split(self, 
                     n_samples: int,
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1) -> Dict[str, np.ndarray]:
        """策略1: 随机划分
        
        Returns:
            dict with 'train', 'val', 'test' indices
        """
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        return {
            'train': indices[:n_train],
            'val': indices[n_train:n_train+n_val],
            'test': indices[n_train+n_val:]
        }
    
    def batch_split(self,
                    batch_labels: np.ndarray,
                    train_batches: List[int],
                    test_batches: List[int],
                    val_ratio: float = 0.1) -> Dict[str, np.ndarray]:
        """策略2: 按批次划分 (域外测试)
        
        args:
            batch_labels: (N,) 批次标签数组
            train_batches: 训练批次列表, e.g. [0, 1, 2]
            test_batches: 测试批次列表, e.g. [3]
            val_ratio: 从训练集划分验证集的比例
            
        Returns:
            dict with 'train', 'val', 'test' indices
        """
        train_mask = np.isin(batch_labels, train_batches)
        test_mask = np.isin(batch_labels, test_batches)
        
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        # 从训练集划分验证集
        if val_ratio > 0:
            train_indices, val_indices = train_test_split(
                train_indices, test_size=val_ratio, random_state=self.random_seed
            )
        else:
            val_indices = np.array([])
        
        return {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
    
    def leave_one_batch_out(self,
                            batch_labels: np.ndarray,
                            test_batch: int,
                            val_ratio: float = 0.1) -> Dict[str, np.ndarray]:
        """策略3: 留一批次交叉验证
        
        Args:
            batch_labels: (N,) 批次标签
            test_batch: 作为测试集的批次ID
            
        Returns:
            dict with 'train', 'val', 'test' indices
        """
        unique_batches = np.unique(batch_labels)
        train_batches = [b for b in unique_batches if b != test_batch]
        
        return self.batch_split(
            batch_labels, train_batches, [test_batch], val_ratio
        )
    
    def get_all_batch_cv_splits(self,
                                batch_labels: np.ndarray,
                                val_ratio: float = 0.1) -> List[Dict[str, np.ndarray]]:
        """获取所有批次交叉验证的划分
        
        Returns:
            list of split dicts, one for each batch as test set
        """
        unique_batches = np.unique(batch_labels)
        splits = []
        
        for test_batch in unique_batches:
            split = self.leave_one_batch_out(
                batch_labels, test_batch, val_ratio
            )
            split['test_batch'] = test_batch
            splits.append(split)
        
        return splits
    
    def stratified_split(self,
                        labels: np.ndarray,
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1) -> Dict[str, np.ndarray]:
        """分层随机划分 (保持类别比例)
        
        Args:
            labels: (N,) 用于分层的标签
        """
        indices = np.arange(len(labels))
        
        # 先划分训练+验证 vs 测试
        test_ratio = 1 - train_ratio - val_ratio
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_ratio, stratify=labels, 
            random_state=self.random_seed
        )
        
        # 再划分训练 vs 验证
        relative_val_ratio = val_ratio / (train_ratio + val_ratio)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=relative_val_ratio, 
            stratify=labels[train_val_idx],
            random_state=self.random_seed
        )
        
        return {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }


def create_experiment_splits(batch_labels: np.ndarray, 
                            n_samples: int) -> Dict[str, Dict]:
    """创建所有实验所需的数据划分
    
    Returns:
        dict mapping experiment_id to split_dict
    """
    splitter = DomainSplitter(random_seed=42)
    
    experiments = {}
    
    # E1-E4: 随机划分
    random_split = splitter.random_split(n_samples)
    experiments['E1'] = {'split': random_split, 'augment': None}
    experiments['E2'] = {'split': random_split, 'augment': 'noise'}
    experiments['E3'] = {'split': random_split, 'augment': 'vae_no_physics'}
    experiments['E4'] = {'split': random_split, 'augment': 'cvae_physics'}
    
    # E5-E6: 批次01-03训练, 04测试
    batch_split = splitter.batch_split(batch_labels, [0, 1, 2], [3])
    experiments['E5'] = {'split': batch_split, 'augment': None}
    experiments['E6'] = {'split': batch_split, 'augment': 'cvae_physics'}
    
    # E7: 留一批次交叉验证
    cv_splits = splitter.get_all_batch_cv_splits(batch_labels)
    experiments['E7'] = {'split': cv_splits, 'augment': 'cvae_physics', 'cv': True}
    
    return experiments


if __name__ == "__main__":
    # 测试划分
    n = 2800
    batch_labels = np.repeat([0, 1, 2, 3], n // 4)
    
    splitter = DomainSplitter()
    
    # 测试随机划分
    split = splitter.random_split(n)
    print(f"随机划分: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
    
    # 测试批次划分
    split = splitter.batch_split(batch_labels, [0, 1, 2], [3])
    print(f"批次划分: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
    
    # 测试CV
    cv_splits = splitter.get_all_batch_cv_splits(batch_labels)
    print(f"CV折数: {len(cv_splits)}")
