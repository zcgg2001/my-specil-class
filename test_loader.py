"""测试数据加载"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import load_all_data

print("开始加载数据...")
spectra, labels = load_all_data()
print(f"光谱形状: {spectra.shape}")
print(f"品牌分布: [hongmei={sum(labels['brand']==0)}, yeshu={sum(labels['brand']==1)}]")
print(f"香料含量: {sorted(set(labels['spice_content']))}")
print(f"批次分布: {[sum(labels['batch']==i) for i in range(4)]}")
print("数据加载测试通过!")
