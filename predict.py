# 用于加载模型和预测

import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from peft import PeftModel, LoraConfig

# 配置参数（根据实际路径修改）
MODEL_PATH = "./bert-base-chinese"  # 基础模型路径
LORA_CHECKPOINT = "./lora_weights/bert"  # LoRA权重路径
MAX_LENGTH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载基础模型和分词器
print("加载基础模型...")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
base_model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2).to(DEVICE)

# 2. 加载LoRA配置和权重
print("加载LoRA配置...")
lora_config = LoraConfig(
    task_type="SEQ_CLS",
    target_modules=["query", "key", "value"],
    inference_mode=True,
    r=8,  # 根据训练时的参数调整
    lora_alpha=16,
    lora_dropout=0.1
)

# 3. 组合基础模型和LoRA
print("加载LoRA权重...")
model = PeftModel.from_pretrained(base_model, LORA_CHECKPOINT, config=lora_config)
model.eval()

def predict(texts):
    """
    预测函数（0-非垃圾短信，1-垃圾短信）
    :param texts: 单条文本或文本列表
    :return: 预测结果列表
    """
    # 处理单条输入
    if isinstance(texts, str):
        texts = [texts]
    
    # 分词处理
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)
    
    # 模型推理
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 结果解析
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1).cpu().numpy().tolist()
    
    return preds[0] if len(preds) == 1 else preds

# 预测示例
if __name__ == "__main__":
    examples = [
        "您的账户已获得100元现金奖励",
        "明天下午三点开会，请准时参加",
        "点击链接领取专属福利",
        "您的年会险即将到，将自动扣费，如需取消请咨",
        "平台會员您好你開通的VIP，将劃 出,1288单号请攞消請詳詢： Route 171",
        "最新消息，【中*京*商*品】即日起受理委托开始办理退款登记。点击→:https://ww.yyhtyy.com"
    ]
    
    results = predict(examples)
    for text, label in zip(examples, results):
        print(f"文本：{text}\n预测结果：{label} ({'垃圾短信' if label == 1 else '非垃圾短信'})")