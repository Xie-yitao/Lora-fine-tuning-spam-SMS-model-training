from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import torch
import model_config_qwen  # 自定义配置文件，需根据实际情况修改

# 加载模型配置
peft_model_id = model_config_qwen.model_path_lora  # LoRA 权重保存路径
config = PeftConfig.from_pretrained(peft_model_id)

# 初始化 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)

# 加载基础模型
model = AutoModelForSequenceClassification.from_pretrained(
    config.base_model_name_or_path,
    num_labels=model_config_qwen.num_labels,
    id2label={"0": "非垃圾短信", "1": "垃圾短信"},
    label2id={"非垃圾短信": 0, "垃圾短信": 1},
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(model, peft_model_id)

# 切换到推理模式
model.eval()

# 推理函数
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    return model.config.id2label[str(predicted_class_id)]  # 确保键是字符串类型

# 测试推理
text = "这是一个测试文本，判断是否为垃圾短信。"
result = predict(text)
print(f"预测结果: {result}")