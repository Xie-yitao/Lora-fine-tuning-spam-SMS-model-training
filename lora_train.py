import os
import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import model_config
from peft import get_peft_model, LoraConfig, TaskType
import time

# 禁用tokenizers并行处理警告 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 加载数据
train_data = pd.read_excel(model_config.train_data_path)
print("="*100)
print(train_data.columns)
train_texts = train_data[train_data.columns[0]].tolist()
train_labels = train_data[train_data.columns[1]].tolist()

dev_data = pd.read_excel(model_config.dev_data_path)
eval_texts = dev_data[dev_data.columns[0]].tolist()
eval_labels = dev_data[dev_data.columns[1]].tolist()

# 初始化模型和分词器
model_name = model_config.model_name_tokenizer_path
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(set(train_labels)))

# 文本编码
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
eval_encodings = tokenizer(eval_texts, truncation=True, padding=True, max_length=64)

# 数据集类
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_encodings, train_labels)
eval_dataset = TextDataset(eval_encodings, eval_labels)

# 更新参数名称并优化训练配置 
training_args = TrainingArguments(
    output_dir=model_config.output_dir,
    logging_dir=model_config.logging_dir,
    save_strategy=model_config.save_strategy,
    save_total_limit=model_config.save_total_limit,
    eval_strategy=model_config.evaluation_strategy,  
    fp16=model_config.use_fp16,
    save_steps=model_config.save_steps,
    eval_steps=model_config.eval_steps,
    load_best_model_at_end=model_config.load_best_model_at_end,
    num_train_epochs=model_config.num_train_epochs,
    per_device_train_batch_size=model_config.train_batch_size,
    per_device_eval_batch_size=model_config.eval_batch_size,
    warmup_steps=model_config.warmup_steps,
    weight_decay=model_config.weight_decay,
    gradient_accumulation_steps=model_config.gradient_accumulation_steps,
    dataloader_drop_last=model_config.dataloader_drop_last,
    dataloader_num_workers=model_config.dataloader_num_workers,
    # 显式声明标签字段
    label_names=["labels"],
)

# LoRA配置
config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    target_modules=["query", "key", "value"],
    inference_mode=False,
    r=model_config.lora_r,
    lora_alpha=model_config.lora_alpha,
    lora_dropout=model_config.lora_dropout,
)
model = get_peft_model(model, config)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 创建保存目录（如果不存在） 
os.makedirs(model_config.output_dir, exist_ok=True)
os.makedirs(model_config.model_path_lora, exist_ok=True)

# 执行训练
start_time = time.time()
trainer.train()

# 显式保存LoRA权重到指定路径 
model.save_pretrained(model_config.model_path_lora)
print(f"LoRA weights saved to: {model_config.model_path_lora}")

end_time = time.time()
print(f"Total training time: {end_time - start_time:.2f} seconds")