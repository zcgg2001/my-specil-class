"""
配置文件: 超参数与路径设置
"""
import os

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULT_DIR = os.path.join(BASE_DIR, "results")

# =============================================================================
# 数据配置
# =============================================================================
WAVELENGTH_START = 200      # nm
WAVELENGTH_END = 1760       # nm
WAVELENGTH_STEP = 5         # nm
NUM_WAVELENGTHS = 313       # 维度

BRANDS = ["hongmei", "yeshu"]
SPICE_LEVELS = [0.0, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6]
BATCHES = ["01", "02", "03", "04"]

# =============================================================================
# 模型配置
# =============================================================================
# cVAE结构
LATENT_DIM = 64
HIDDEN_DIMS = [512, 256]
CONDITION_DIM = 3  # brand(1) + spice_content(1) + batch(1, optional)

# 损失函数权重
BETA_KL = 0.1           # KL散度权重
LAMBDA_SMOOTH = 1.0     # 平滑性约束
LAMBDA_RANGE = 10.0     # 范围约束
LAMBDA_ATTR = 5.0       # 属性一致性约束

# =============================================================================
# 训练配置
# =============================================================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS_PREDICTOR = 50   # 香料预测器
EPOCHS_CVAE = 100       # cVAE
EPOCHS_MULTITASK = 100  # 多任务模型

# 数据增强
AUGMENT_RATIO_MAX = 3.0  # 最大增强倍数
AUGMENT_MINORITY_ONLY = True  # 仅增强少数类

# =============================================================================
# 实验配置
# =============================================================================
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
