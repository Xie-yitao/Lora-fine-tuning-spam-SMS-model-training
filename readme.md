### 算法功能

微调大模型（qwen）/bert，对垃圾短信分类，对输入的短信进行分类，识别垃圾短信与非垃圾短信，输出0表示非垃圾短信，输出1表示垃圾短信。（二分类）

### 环境安装
python版本3.12.9
```
conda creaate -n <你的项目名称> python=3.12
pip install -r requirements.txt
```

### 文件结构
```
bert-base-chinese           #如果使用bert，见bert下载至改文件夹
qwen-7b-hf                  #如果使用7b，见qwen模型下载至改文件夹
dataset
|-dataset
  |-dev_data.xlsx
  |-test_data.xlsx
  |-train_dataset.xlsx
|-Original_dataset
  |-垃圾短信分类数据.xlsx（可以替换为你的数据，可以多个表格文件）
img
|-bert
|-qwen
lora_results
|-bert
|-qwen
lora_weights
|-bert
|-qwen
dataset.py
lora_train.py
lora_test.py
lora_train_qwen.py  #使用hf模型训练微调qwen，本次使用qwen0.5b，可以自行修改
model_config_qwen.py
predict.py
lora_train_qwen.py
lora_test_qwen.py
requirements.txt
readme.md
```
### 核心代码(以微调bert为例)：

```
1、数据处理脚本：dataset.py                 #数据处理脚本
2、训练脚本：lora_train.py                  #模型训练(bert)
2、测试脚本：lora_test.py                   #输出测试集的混淆矩阵图片保存在img目录下(bert)
3、推理脚本：predict.py                     #根据需要修改，目前没封装，只做了模型载入与推理(bert)
4、配置脚本：model_config.py                #可以更换为yml配置，对应的修改测试脚本和推理脚本即可(需要重写编写yml配置脚本与导入yaml包,bert)
5、数据喝茶脚本：make_data.py               #生成训练与测试数据（使用kimi生成，数据质量与标签不准，仅提代码流程跑通测试）
6、qwen训练脚本：lora_qwen_cls.py           #微调qwen
7、qwen推理脚本：lora_qwen_cls_pre.py       #qwen推理脚本
8、qwen测试、训练与推理:qwen_lora.ipynb     #jupyter版本，方便快速测试，包含模型加载，训练与推理所有完整代码
```

### 其他文件

```
dataset #数据集
img #存储推理混淆矩阵结果
```

### 运行准备
下载对应模型到指定文件夹，如bert-base-chinese、qwen-7b-hf，如果可以科学上网，参考qwen_lora.ipynb里边设置代理，也可以指定下载路径

```
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 设置代理（替换为你的实际代理地址）
proxies = {
    "http": "http://127.0.0.1:7890",   # 例如：http://127.0.0.1:7890
    "https": "http://127.0.0.1:7890",  # 例如：http://127.0.0.1:7890
}
model_name = "qwen/Qwen2-0.5B-Instruct"
# 指定下载路径
download_dir = "./models"  # 指定下载路径
# 下载并加载模型和分词器（显式传递代理）
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    proxies=proxies,
    cache_dir=download_dir
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    proxies=proxies,
    cache_dir=download_dir
)
print(model)
```

## 微调qwen
### 1、数据准备
1. 按照dataset/Original_dataset/短信分类数据.xlsx的文件结构进行存储训练数据，将自己的数据放到"dataset/Original_dataset/"目录底下，运行make_daata.py代码生成示例数据
2. 运行dataset.py对数据进行划分
3. 运行后文件结构如下：
dataset
|-Original_dataset
  |-垃圾短信分类数据.xlsx（可以替换为你的数据，可以多个表格文件）
|-dataset
  |-dev_data.xlsx
  |-test_data.xlsx
  |-train_data.xlsx

### 2、可以直接参考qwen_lora.ipynb
