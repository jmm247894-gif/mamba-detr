
from detrex.config import get_config

from .models.deformable_detr_mamba_T import model
from detectron2.data.datasets import register_coco_instances

# ==============================================================================
# 2. 数据集注册 (保持不变)
# ==============================================================================
register_coco_instances(
    name="my_coco_2017_train",
    metadata={},
    json_file="/mnt/d/Python/data/coco/annotations/instances_train2017.json",
    image_root="/mnt/d/Python/data/coco/train2017/"
)
register_coco_instances(
    name="my_coco_2017_val",
    metadata={},
    json_file="/mnt/d/Python/data/coco/annotations/instances_val2017.json",
    image_root="/mnt/d/Python/data/coco/val2017/"
)

# --- 后续配置基本保持不变 ---
dataloader = get_config("common/data/coco_detr.py").dataloader
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# -----------------------------------------------------------------------------------------
# CHANGE 2: 修改检查点和输出目录
# -----------------------------------------------------------------------------------------
# Mamba的预训练权重是在 build_mamba_vision_backbone 函数内部加载的，
# 所以我们不再需要这个指向 ResNet 权重的路径。
# train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl" # <-- 注释或删除此行
train.output_dir = "./output/deformable_detr_mamba_T_12ep" # <-- 修改输出目录

# max training iterations
train.max_iter = 709720

# run evaluation every 5000 iters
train.eval_period = 59144

# log training infomation every 20 iters
train.log_period = 500

# save checkpoint every 5000 iters
train.checkpointer.period = 177432

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# -----------------------------------------------------------------------------------------
# CHANGE 3 (可选但推荐): 调整优化器设置
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# MambaVision的参数名可能不包含 "backbone"，但我们的封装类保证了它在 `mamba_model` 之下。
# 为了安全起见，我们可以调整学习率衰减的逻辑。
# 不过，通常直接的 "backbone" 检查也能工作，因为Detrex会将构建的模型赋给 `model.backbone`。
# 我们暂时保持不变，如果训练时 backbone 学习率没有降低，再来修改。
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1


# --- 数据加载配置 ---
dataloader.train.dataset.names = "my_coco_2017_train"
dataloader.test.dataset.names = "my_coco_2017_val"
dataloader.evaluator.dataset_name = "my_coco_2017_val"
dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 2
dataloader.evaluator.output_dir = train.output_dir