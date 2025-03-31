# import torch
# import pandas as pd
# from datasets import Dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     AutoConfig,
#     TrainingArguments,
#     Trainer,
#     EarlyStoppingCallback
# )
# from peft import (
#     get_peft_model,
#     LoraConfig,
#     TaskType
# )
# import transformers
# import numpy as np
# from sklearn.metrics import accuracy_score
# from torch.nn import Linear, CrossEntropyLoss
# from transformers.utils import ModelOutput
# from typing import Optional, Tuple
# from dataclasses import dataclass
# import model_config_qwen  # 自定义配置文件
# import os

# # 环境配置
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# os.environ["TRUST_REMOTE_CODE"] = "true"

# # ========== 配置类 ==========
# class ModelConfig:
#     model_path = model_config_qwen.model_name_tokenizer_path               # 模型路径
#     num_labels = model_config_qwen.num_labels                              # 分类类别数
#     model_min_length = model_config_qwen.model_min_length                  # 最小输入长度
#     freeze_base_model = model_config_qwen.freeze_base_model                # 冻结基础模型参数
#     pooling_type = model_config_qwen.pooling_type                          # [last, first, mean] 池化方式

# class TrainingConfig:
#     output_dir = model_config_qwen.output_dir                              # 输出目录
#     max_epochs = model_config_qwen.max_epochs                              # 最大训练轮次
#     batch_size = model_config_qwen.train_batch_size                        # 训练批次大小(V100 32G可调整到24-32)
#     eval_batch_size = model_config_qwen.test_batch_size                    # 验证批次大小
#     learning_rate = model_config_qwen.learning_rate                        # 学习率
#     weight_decay = model_config_qwen.weight_decay                          # 权重衰减
#     gradient_accumulation = model_config_qwen.gradient_accumulation_steps  # 梯度累积步数
#     fp16 = model_config_qwen.use_fp16                                      # 启用混合精度
#     gradient_checkpointing = model_config_qwen.gradient_checkpointing      # 梯度检查点
#     early_stop_patience = model_config_qwen.early_stop_patience            # 早停耐心值

# class LoRAConfig:
#     r = model_config_qwen.lora_r                                           # LoRA秩
#     lora_alpha = model_config_qwen.lora_alpha                              # LoRA alpha值
#     lora_dropout = model_config_qwen.lora_dropout                          # Dropout率
#     target_modules = model_config_qwen.target_modules                      # 目标模块

# # ========== 数据类定义 ==========
# @dataclass
# class SequenceClassifierOutput(ModelOutput):
#     loss: Optional[torch.FloatTensor] = None
#     logits: torch.FloatTensor = None
#     hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     attentions: Optional[Tuple[torch.FloatTensor]] = None

# # ========== 模型类 ==========
# class QwenForSequenceClassification(transformers.PreTrainedModel):
#     supports_gradient_checkpointing = True  # 新增
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        
#         # 分类头
#         self.classifier = Linear(config.hidden_size, config.num_labels)
#         self.pooling_type = ModelConfig.pooling_type
        
#         # 冻结基础模型
#         if ModelConfig.freeze_base_model:
#             for param in self.model.parameters():
#                 param.requires_grad = False
        
#         self.post_init()

#     def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
#         outputs = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             output_hidden_states=True,
#             use_cache=False  # 禁用缓存节省显存
#         )
        
#         # 特征池化
#         last_hidden = outputs.hidden_states[-1]
#         if self.pooling_type == "first":
#             pooled = last_hidden[:, 0, :]
#         elif self.pooling_type == "mean":
#             pooled = last_hidden.mean(dim=1)
#         else:  # last
#             pooled = last_hidden[torch.arange(last_hidden.size(0)), (attention_mask != 0).sum(dim=1)-1]
        
#         logits = self.classifier(pooled)
#         loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions
#         )

# # ========== 工具函数 ==========
# def analyze_text_length(texts, tokenizer):
#     """分析文本长度分布"""
#     lengths = []
#     for text in texts:
#         tokens = tokenizer.tokenize(text)
#         lengths.append(len(tokens))
    
#     print(f"\n📊 文本长度分析:")
#     print(f"最小: {min(lengths)}, 最大: {max(lengths)}, 平均: {np.mean(lengths):.1f}")
#     print(f"75%分位数: {np.percentile(lengths, 75)}, 95%分位数: {np.percentile(lengths, 95)}")
    
#     # 动态确定max_length
#     max_length = min(
#         int(np.percentile(lengths, 95)), 
#         tokenizer.model_max_length
#     )
#     max_length = max(max_length, ModelConfig.model_min_length)
#     print(f"推荐max_length: {max_length}")
#     return max_length

# def prepare_dataset(tokenizer, texts, labels, max_length):
#     """数据集预处理"""
#     dataset = Dataset.from_dict({"text": texts, "label": labels})
    
#     def tokenize_fn(examples):
#         tokenized = tokenizer(
#             examples["text"],
#             padding="max_length",
#             truncation=True,
#             max_length=max_length,
#             return_tensors="pt",
#         )
#         return {
#             "input_ids": tokenized["input_ids"][0],
#             "attention_mask": tokenized["attention_mask"][0],
#             "label": examples["label"]
#         }
    
#     return dataset.map(
#         tokenize_fn,
#         batched=False,
#         remove_columns=["text"],
#         num_proc=4,
#         desc="Tokenizing"
#     )

# # ========== 主流程 ==========
# def main():
#     # 硬件检查
#     print(f"\n🖥️ 硬件配置检查:")
#     print(f"PyTorch版本: {torch.__version__}")
#     print(f"CUDA可用: {torch.cuda.is_available()}")
#     if torch.cuda.is_available():
#         print(f"GPU: {torch.cuda.get_device_name(0)}")
#         print(f"显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

#     # 初始化配置
#     model_config = ModelConfig()
#     train_config = TrainingConfig()
#     lora_config = LoRAConfig()

#     # 加载数据
#     print("\n📂 加载数据集...")
#     train_df = pd.read_excel(model_config_qwen.train_data_path)
#     valid_df = pd.read_excel(model_config_qwen.dev_data_path)
#     print(f"训练集: {len(train_df)}条, 验证集: {len(valid_df)}条")

#     # 初始化Tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_config.model_path,
#         trust_remote_code=True,
#         pad_token="<|endoftext|>"
#     )

#     # 动态确定max_length
#     max_length = analyze_text_length(
#         train_df.iloc[:, 0].tolist(),
#         tokenizer
#     )

#     # 准备数据集
#     train_dataset = prepare_dataset(
#         tokenizer,
#         train_df.iloc[:, 0].tolist(),
#         train_df.iloc[:, 1].tolist(),
#         max_length
#     )
#     valid_dataset = prepare_dataset(
#         tokenizer,
#         valid_df.iloc[:, 0].tolist(),
#         valid_df.iloc[:, 1].tolist(),
#         max_length
#     )

#     # 加载基础模型
#     print("\n🚀 加载预训练模型...")
#     config = AutoConfig.from_pretrained(
#         model_config.model_path,
#         num_labels=model_config.num_labels,
#         trust_remote_code=True
#     )
#     model = QwenForSequenceClassification(config)
    
#     # 应用LoRA
#     print("\n🎛️ 应用LoRA配置...")
#     peft_config = LoraConfig(
#         r=lora_config.r,
#         lora_alpha=lora_config.lora_alpha,
#         lora_dropout=lora_config.lora_dropout,
#         target_modules=lora_config.target_modules,
#         task_type=TaskType.SEQ_CLS,
#         inference_mode=False,
#         bias="none"
#     )
#     model = get_peft_model(model, peft_config)
#     model.print_trainable_parameters()

#     # 训练参数配置
#     training_args = TrainingArguments(
#         output_dir=train_config.output_dir,
#         evaluation_strategy="epoch",
#         save_strategy="epoch",
#         learning_rate=train_config.learning_rate,
#         per_device_train_batch_size=train_config.batch_size,
#         per_device_eval_batch_size=train_config.eval_batch_size,
#         gradient_accumulation_steps=train_config.gradient_accumulation,
#         num_train_epochs=train_config.max_epochs,
#         weight_decay=train_config.weight_decay,
#         fp16=train_config.fp16,
#         gradient_checkpointing=train_config.gradient_checkpointing,
#         load_best_model_at_end=True,
#         metric_for_best_model="accuracy",
#         logging_steps=50,
#         save_total_limit=2,
#         report_to="none",
#         optim="adamw_torch",
#         dataloader_num_workers=4,
#         lr_scheduler_type="cosine",
#         warmup_ratio=0.1
#     )

#     # 评估函数
#     def compute_metrics(p):
#         preds = np.argmax(p.predictions, axis=1)
#         return {"accuracy": accuracy_score(p.label_ids, preds)}

#     # 训练器
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=valid_dataset,
#         compute_metrics=compute_metrics,
#         callbacks=[
#             EarlyStoppingCallback(early_stopping_patience=train_config.early_stop_patience)
#         ]
#     )

#     # 开始训练
#     print("\n🚂 启动训练...")
#     train_result = trainer.train()
    
#     # 保存模型
#     print("\n💾 保存最佳模型...")
#     model.save_pretrained(os.path.join(train_config.output_dir, "best_model"))
#     tokenizer.save_pretrained(os.path.join(train_config.output_dir, "best_model"))
#     # 显式保存LoRA权重到指定路径 
#     model.save_pretrained(model_config_qwen.model_path_lora)
#     print(f"LoRA weights saved to: {model_config_qwen.model_path_lora}")

# if __name__ == "__main__":
#     # 显存优化配置
#     torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32矩阵运算
#     torch.backends.cudnn.benchmark = True        # 启用cuDNN基准优化
#     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 减少显存碎片
    
#     main()



import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType
)
import transformers
import numpy as np
from sklearn.metrics import accuracy_score
from torch.nn import Linear, CrossEntropyLoss
from transformers.utils import ModelOutput
from typing import Optional, Tuple
from dataclasses import dataclass
import model_config_qwen  # 自定义配置文件
import os

# 环境配置
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "true"

# ========== 配置类 ==========
class ModelConfig:
    model_path = model_config_qwen.model_name_tokenizer_path               
    num_labels = model_config_qwen.num_labels                              
    model_min_length = model_config_qwen.model_min_length                  
    freeze_base_model = model_config_qwen.freeze_base_model                
    pooling_type = model_config_qwen.pooling_type                          # [last, first, mean]

class TrainingConfig:
    output_dir = model_config_qwen.output_dir                              
    max_epochs = model_config_qwen.max_epochs                              
    batch_size = model_config_qwen.train_batch_size                        
    eval_batch_size = model_config_qwen.test_batch_size                    
    learning_rate = model_config_qwen.learning_rate                        
    weight_decay = model_config_qwen.weight_decay                          
    gradient_accumulation = model_config_qwen.gradient_accumulation_steps  
    fp16 = model_config_qwen.use_fp16                                      
    gradient_checkpointing = model_config_qwen.gradient_checkpointing      
    early_stop_patience = model_config_qwen.early_stop_patience            

class LoRAConfig:
    r = model_config_qwen.lora_r                                           
    lora_alpha = model_config_qwen.lora_alpha                              
    lora_dropout = model_config_qwen.lora_dropout                          
    target_modules = model_config_qwen.target_modules                      

# ========== 数据类定义 ==========
@dataclass
class SequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

# ========== 模型类 ==========
class QwenForSequenceClassification(transformers.PreTrainedModel):
    supports_gradient_checkpointing = True
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # 关键修复：显式设置模型类型
        config.model_type = "qwen"
        config.architectures = ["QwenForSequenceClassification"]
        
        # 初始化基础模型
        self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        self.classifier = Linear(config.hidden_size, config.num_labels)
        self.pooling_type = ModelConfig.pooling_type
        
        # 冻结基础模型参数
        if ModelConfig.freeze_base_model:
            for param in self.model.parameters():
                param.requires_grad = False
                
        self.post_init()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, AutoModelForCausalLM):
            module.gradient_checkpointing = value

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 显式禁用缓存以节省显存
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False  # 关键参数
        )
        
        # 优化特征池化
        last_hidden = outputs.hidden_states[-1]
        if self.pooling_type == "first":
            pooled = last_hidden[:, 0, :]
        elif self.pooling_type == "mean":
            pooled = last_hidden.mean(dim=1)
        else:
            pooled = last_hidden[torch.arange(last_hidden.size(0)), (attention_mask != 0).sum(dim=1)-1]
        
        # 分类头
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

# ========== 工具函数 ==========
def analyze_text_length(texts, tokenizer):
    """动态确定max_length"""
    lengths = [len(tokenizer.tokenize(text)) for text in texts]
    print(f"\n📊 文本长度分析（95%分位数: {np.percentile(lengths, 95):.0f}）")
    return max(
        min(int(np.percentile(lengths, 95)), 
        tokenizer.model_max_length
    ), ModelConfig.model_min_length)

def prepare_dataset(tokenizer, texts, labels, max_length):
    """带显存优化的数据预处理"""
    dataset = Dataset.from_dict({"text": texts, "label": labels})
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    
    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        num_proc=4,
        desc="Tokenizing",
        batch_size=100  # 批处理提升处理速度
    )

# ========== 主流程 ==========
def main():
    # 硬件检查
    print(f"\n🖥️ 硬件检查:")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

    # 初始化配置
    model_config = ModelConfig()
    train_config = TrainingConfig()
    lora_config = LoRAConfig()

    # 数据加载
    print("\n📂 加载数据集...")
    train_df = pd.read_excel(model_config_qwen.train_data_path)
    valid_df = pd.read_excel(model_config_qwen.dev_data_path)
    print(f"训练集: {len(train_df)}条 | 验证集: {len(valid_df)}条")

    # 初始化Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_path,
        trust_remote_code=True,
        pad_token="<|endoftext|>"
    )

    # 动态确定max_length
    max_length = analyze_text_length(train_df.iloc[:, 0].tolist(), tokenizer)
    print(f"使用max_length: {max_length}")

    # 准备数据集
    train_dataset = prepare_dataset(tokenizer, train_df.iloc[:, 0].tolist(), train_df.iloc[:, 1].tolist(), max_length)
    valid_dataset = prepare_dataset(tokenizer, valid_df.iloc[:, 0].tolist(), valid_df.iloc[:, 1].tolist(), max_length)

    # 模型配置
    config = AutoConfig.from_pretrained(
        model_config.model_path,
        num_labels=model_config.num_labels,
        trust_remote_code=True
    )
    # 关键修复：确保保存的配置包含模型类型
    config.model_type = "qwen"
    config.architectures = ["QwenForSequenceClassification"]

    # 初始化模型
    print("\n🚀 初始化模型...")
    model = QwenForSequenceClassification(config)
    
    # 应用LoRA
    print("\n🎛️ 应用LoRA配置...")
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 训练参数
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        evaluation_strategy="epoch",
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

    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))},
        callbacks=[EarlyStoppingCallback(early_stopping_patience=train_config.early_stop_patience)]
    )

    # 开始训练
    print("\n🚂 启动训练...")
    trainer.train()
    
    # 保存模型（关键修改）
    print("\n💾 保存模型...")
    final_model_path = os.path.join(train_config.output_dir, "best_model")
    
    # 保存完整模型配置
    config.save_pretrained(final_model_path)
    # 保存适配器权重
    model.save_pretrained(final_model_path)
    # 保存Tokenizer
    tokenizer.save_pretrained(final_model_path)
    
    print(f"模型已保存至：{final_model_path}")

if __name__ == "__main__":
    # 显存优化配置
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    main()