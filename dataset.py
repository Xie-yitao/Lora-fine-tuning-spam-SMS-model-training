import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import os

### 文件路径配置
input_directory = './dataset/Original_dataset'  # 输入文件夹路径
output_directory = './dataset/dataset'  # 输出文件夹路径
print("开始读取文件")
### 批量读取Excel文件
excel_files = [
    os.path.join(input_directory, filename)
    for filename in os.listdir(input_directory)
    if filename.endswith(('.xlsx', '.xls'))  # 支持多种格式[[10]]
]
print("开始处理文件")

# 合并所有Excel文件
dataframes = []
for file_path in excel_files:
    df = pd.read_excel(file_path, engine='openpyxl')  # 显式指定引擎[[10]]
    dataframes.append(df)
merged_data = pd.concat(dataframes, ignore_index=True)

### 数据集划分（6:2:2）
# 划分测试集（20%）
train_val_data, test_set = train_test_split(merged_data, test_size=0.2, random_state=42)

# 划分训练集和验证集（各占剩余80%的50%）
training_set, validation_set = train_test_split(train_val_data, test_size=0.25, random_state=42)

### 保存数据集
os.makedirs(output_directory, exist_ok=True)  # 创建输出目录
print("文件处理结束")
# 分别保存三个数据集
training_set.to_excel(os.path.join(output_directory, 'train_data.xlsx'), index=False)
validation_set.to_excel(os.path.join(output_directory, 'dev_data.xlsx'), index=False)
test_set.to_excel(os.path.join(output_directory, 'test_data.xlsx'), index=False)  
print("文件已经保存到/dataset/dataset里边")