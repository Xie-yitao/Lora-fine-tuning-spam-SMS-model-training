### 算法功能

垃圾短信分类，对输入的短信进行分类，识别垃圾短信与非垃圾短信，输出0表示非垃圾短信，输出1表示垃圾短信。（二分类）

### 环境安装

```
conda env create -f requirements.yml
```

### 文件结构
```
bert-base-chinese
qwen-7b-hf
dataset
|-dataset
  |-dev_data.xlsx
  |-test_data.xlsx
  |-train_dataset.xlsx
|-Original_dataset
  |-dataset_2024.xlsx
  |-dataset_20250211.xlsx
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
lora_train_qwen.py  #使用hf模型，并自定义分类头类别，需要注意模型保存以及载入方法
model_config_qwen.py
predict.py
lora_train_qwen.py
lora_test_qwen.py
requirements.yml
readme.md
```
### 核心代码(以微调bert为例)：

```
1、数据处理脚本：dataset.py #数据处理脚本
2、训练脚本：lora_train.py #模型训练(bert)
2、测试脚本：lora_test.py #输出测试集的混淆矩阵图片保存在img目录下(bert)
3、推理脚本：predict.py #根据需要修改，目前没封装，只做了模型载入与推理(bert)
4、配置脚本：model_config.py #可以更换为yml配置，对应的修改测试脚本和推理脚本即可(需要重写编写yml配置脚本与导入yaml包,bert)
```

### 其他文件

```
dataset #数据集
img #存储推理混淆矩阵结果
```

### 运行准备
下载对应模型到指定文件夹，如bert-base-chinese、qwen-7b-hf
