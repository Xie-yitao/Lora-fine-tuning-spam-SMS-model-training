import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType
)
import numpy as np
from sklearn.metrics import accuracy_score
import model_config_qwen  # 自定义配置文件，需根据实际情况修改
import os

# 环境配置
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "true"
model_name = "qwen/Qwen2-0.5B-Instruct"
# 代理设置（根据需要配置）
proxies = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}

# ========== 配置类 ==========
class ModelConfig:
    model_path = model_config_qwen.model_name_tokenizer_path   # 模型保存路径
    num_labels = model_config_qwen.num_labels                 # 分类类别数
    model_min_length = model_config_qwen.model_min_length     # 最小输入长度
    freeze_base_model = model_config_qwen.freeze_base_model   # 是否冻结基础模型参数（只训练 LoRA 参数）
    pooling_type = model_config_qwen.pooling_type             # 池化方式（如 last, first, mean）

class TrainingConfig:
    output_dir = model_config_qwen.output_dir                 # 输出目录
    max_epochs = model_config_qwen.max_epochs                 # 最大训练轮次
    batch_size = model_config_qwen.train_batch_size           # 训练批次大小
    eval_batch_size = model_config_qwen.test_batch_size         # 验证批次大小
    learning_rate = model_config_qwen.learning_rate           # 学习率
    weight_decay = model_config_qwen.weight_decay             # 权重衰减
    gradient_accumulation = model_config_qwen.gradient_accumulation_steps  # 梯度累积步数
    fp16 = model_config_qwen.use_fp16                         # 是否启用混合精度
    gradient_checkpointing = model_config_qwen.gradient_checkpointing  # 是否启用梯度检查点
    early_stop_patience = model_config_qwen.early_stop_patience           # 早停耐心值

class LoRAConfig:
    r = model_config_qwen.lora_r                  # LoRA 秩
    lora_alpha = model_config_qwen.lora_alpha     # LoRA alpha 值
    lora_dropout = model_config_qwen.lora_dropout # LoRA dropout 率
    target_modules = model_config_qwen.target_modules  # LoRA 所针对的目标模块列表

# ========== 工具函数 ==========
def analyze_text_length(texts, tokenizer):
    """分析文本长度分布，并动态确定 max_length"""
    lengths = []
    for text in texts:
        tokens = tokenizer.tokenize(text)
        lengths.append(len(tokens))
    print(f"\n📊 文本长度分析:")
    print(f"最小: {min(lengths)}, 最大: {max(lengths)}, 平均: {np.mean(lengths):.1f}")
    print(f"75% 分位数: {np.percentile(lengths, 75)}, 95% 分位数: {np.percentile(lengths, 95)}")
    max_length = min(int(np.percentile(lengths, 95)), tokenizer.model_max_length)
    max_length = max(max_length, ModelConfig.model_min_length)
    print(f"推荐 max_length: {max_length}")
    return max_length

def prepare_dataset(tokenizer, texts, labels, max_length):
    """数据集预处理，将文本 token 化，同时保留标签"""
    dataset = Dataset.from_dict({"text": texts, "label": labels})
    
    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "label": examples["label"]
        }
    
    return dataset.map(
        tokenize_fn,
        batched=False,
        remove_columns=["text"],
        num_proc=4,
        desc="Tokenizing"
    )

# ========== 主流程 ==========
def main():
    # 硬件检查
    print(f"\n🖥️ 硬件配置检查:")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    # 初始化配置
    model_config = ModelConfig()
    train_config = TrainingConfig()
    lora_config = LoRAConfig()

    # 加载数据
    print("\n📂 加载数据集...")
    train_df = pd.read_excel(model_config_qwen.train_data_path)
    valid_df = pd.read_excel(model_config_qwen.dev_data_path)
    print(f"训练集: {len(train_df)} 条, 验证集: {len(valid_df)} 条")

    # 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        proxies=proxies,
        pad_token="<|endoftext|>"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 动态确定 max_length
    max_length = analyze_text_length(train_df.iloc[:, 0].tolist(), tokenizer)

    # 准备数据集
    train_dataset = prepare_dataset(
        tokenizer,
        train_df.iloc[:, 0].tolist(),
        train_df.iloc[:, 1].tolist(),
        max_length
    )
    valid_dataset = prepare_dataset(
        tokenizer,
        valid_df.iloc[:, 0].tolist(),
        valid_df.iloc[:, 1].tolist(),
        max_length
    )

    # 加载预训练模型
    print("\n🚀 加载预训练模型...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=model_config.num_labels,
        id2label={"0": "非垃圾短信", "1": "垃圾短信"},
        label2id={"非垃圾短信": 0, "垃圾短信": 1},
        # torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        proxies=proxies,
        use_cache=False  # 禁用缓存以兼容梯度检查点
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # 应用 LoRA 微调配置
    print("\n🎛️ 应用 LoRA 配置...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=lora_config.target_modules,
        inference_mode=False,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        modules_to_save=["score"]  # 保留分类头参数可训练
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 检查可训练参数
    print("\n🔍 可训练参数检查:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"可训练参数: {name}")

    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        eval_strategy="epoch",  # 修正参数名称为evaluation_strategy，报错就改为eval_strategy
        save_strategy="epoch",
        learning_rate=train_config.learning_rate,
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=train_config.eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation,
        num_train_epochs=train_config.max_epochs,
        weight_decay=train_config.weight_decay,
        fp16=train_config.fp16,
        gradient_checkpointing=train_config.gradient_checkpointing,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        optim="adamw_torch",
        dataloader_num_workers=4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1
    )

    # 评估函数（计算准确率）
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": accuracy_score(p.label_ids, preds)}

    # 构建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=train_config.early_stop_patience)]
    )

    # 开始微调训练
    print("\n🚂 启动微调...")
    train_result = trainer.train()

    # 保存最佳模型和 LoRA 权重
    print("\n💾 保存最佳模型...")
    best_model_dir = os.path.join(train_config.output_dir, "best_model")
    model.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    lora_model_dir = model_config_qwen.model_path_lora
    model.save_pretrained(lora_model_dir)
    print(f"LoRA 权重保存至: {lora_model_dir}")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True  # 启用 TF32 矩阵运算
    torch.backends.cudnn.benchmark = True         # 启用 cuDNN 基准优化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 减少显存碎片
    main()


