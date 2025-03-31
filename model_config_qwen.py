# model_config.py

# 模型和数据的路径
model_name_tokenizer_path = "./qwen-7b-hf"                             # 预训练模型和分词器的路径
model_path_lora = "./lora_weights/qwen"                                # LoRA 权重保存路径
train_data_path = "./dataset/dataset/train_dataset.xlsx"               # 训练数据路径
dev_data_path = "./dataset/dataset/dev_data.xlsx"                      # 验证数据路径
test_data_path = "./dataset/dataset/test_data.xlsx"                    # 测试数据路径
output_dir="./lora_results/qwen"
logging_dir="./lora_logs/qwen"
# 模型配置
num_labels = 2                                                         # 分类类别数
model_min_length = 128                                                  # 最小输入长度
freeze_base_model = True                                               # 冻结基础模型参数
pooling_type = "last"                                                  # [last, first, mean] 池化方式
# 训练和测试的超参数
max_epochs = 3                                                         # 最大训练轮次
save_strategy = "steps"                                                # 保存策略，按steps保存
save_total_limit = 1                                                   # 保存模型的最大数量
evaluation_strategy = "steps"                                          # 评估策略，按steps评估
load_best_model_at_end=True                                            # 训练结束后加载在评估过程中表现最好的模型
use_fp16 = True                                                        # 混合精度训练
test_batch_size = 16                                                   # 测试时的批量大小
train_batch_size = 16                                                  # 训练时的批量大小
eval_batch_size = 16                                                  # 验证时的批量大小
max_seq_length = 512                                                   # 文本的最大长度
learning_rate = 2e-4                                                   # 学习率
num_train_epochs = 5                                                   # 训练的轮数
warmup_steps = 1250                                                    # 学习率预热步数
weight_decay = 0.01                                                    # 权重衰减
gradient_accumulation_steps=2                                          # 梯度累加
save_steps = 400                                                       # 每多少步保存一次模型
eval_steps = 200                                                       # 每多少步评估一次模型
dataloader_drop_last = True                                            # 是否丢弃最后一个不完整的
gradient_checkpointing = True                                          # 梯度检查点
early_stop_patience = 3                                                # 早停耐心值
dataloader_num_workers=4                                               # 多线程加载

# LoRA 配置参数
lora_r = 8                                                             # LoRA 的秩
lora_alpha = 32                                                        # LoRA 的alpha值
lora_dropout = 0.05                                                    # LoRA 的dropout率
target_modules = ["c_attn", "w1", "w2", "c_proj"]                      # 目标模块