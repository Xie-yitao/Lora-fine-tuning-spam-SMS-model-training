# 测试LoRA 模型


import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import seaborn as sns
from torch.utils.data import DataLoader
from datasets import Dataset
import matplotlib.pyplot as plt
import model_config
from peft import PeftModel, LoraConfig, TaskType
from tqdm import tqdm

# 加载测试集
print('开始加载数据集')
test_data_path = model_config.test_data_path
test_data = pd.read_excel(test_data_path) 
texts = test_data[test_data.columns[0]].tolist()
labels = test_data[test_data.columns[1]].tolist()
print('数据集载入成功')

# 原始模型
model_path = model_config.model_name_tokenizer_path
model_path_loar = model_config.model_path_lora

tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(set(labels)))

# 加载LoRA配置
config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    target_modules=["query", "key", "value"],
    inference_mode=True,
    r=model_config.lora_r,
    lora_alpha=model_config.lora_alpha,
    lora_dropout=model_config.lora_dropout,
)
print('开始加载模型')
# 加载LoRA权重
model = PeftModel.from_pretrained(model, model_config.model_path_lora, config=config)
print('模型加载成功')

# 将模型设置为评估模式
model.eval()

# 确保模型在 GPU 上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

# 定义数据具体处理逻辑
def collate_fn(batch):
    texts = [item["text"] for item in batch]
    labels = [item["label"] for item in batch]
    encoding = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    encoding["labels"] = torch.tensor(labels)
    return encoding

# 创建Dataset对象
print('开始创建数据集')
dataset = Dataset.from_dict({"text": texts, "label": labels})
data_loader = DataLoader(dataset, batch_size=model_config.test_batch_size, collate_fn=collate_fn, num_workers=8)
print('完成创建数据集')

predictions = []
print('开始推理')


# 使用 tqdm 包裹 data_loader，创建进度条
with tqdm(total=len(data_loader), desc="Testing") as pbar:
    for batch in data_loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        if model.config.num_labels > 2:
            batch_predictions = torch.argmax(logits, dim=1).cpu().tolist()
        else:
            # 修正二分类预测逻辑，确保输出一维列表
            probs = torch.softmax(logits, dim=1)  # 使用softmax处理logits 
            batch_predictions = torch.argmax(probs, dim=1).cpu().tolist()  # 获取类别标签

        predictions.extend(batch_predictions)
        pbar.update(1)

# 确保 labels 和 predictions 的格式一致
labels = [int(label) for label in labels]

# 验证格式一致性（调试用）
print(f"Labels shape: {len(labels)}, Example: {labels[:5]}")
print(f"Predictions shape: {len(predictions)}, Example: {predictions[:5]}")

# 计算准确度、精确度和召回率
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions, average="weighted", zero_division=0)
recall = recall_score(labels, predictions, average="weighted", zero_division=0)

# 输出结果
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# 绘制混淆矩阵
cm = confusion_matrix(labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("img/bert/confusion_matrix_lora.png")
plt.show()