import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取Excel文件并指定数据类型为整数
df = pd.read_excel('output.xlsx', header=None, dtype=int)

# 将DataFrame展平为一维numpy数组
array = df.values.flatten()

# 将numpy数组转换为pandas的Series对象
series = pd.Series(array)

# 计算每个评分出现的频率
frequency = series.value_counts().sort_index()

# 计算每个评分的累积百分率
cumulative_percentages = frequency.cumsum() / frequency.sum()

# 绘制频率分析表
frequency_table = pd.DataFrame({'Frequency': frequency, 'Cumulative Percentages': cumulative_percentages})
print(frequency_table)

# 绘制直方图
plt.figure(figsize=(8, 6))
plt.hist(series, bins=5, range=(1, 6), rwidth=0.8, align='mid')
plt.xlabel('Satisfaction Rating')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Satisfaction Ratings')
plt.xticks(range(1, 6))
plt.show()

# 绘制饼图
plt.figure(figsize=(6, 6))
plt.pie(frequency, labels=frequency.index, autopct='%1.1f%%')
plt.title('Pie Chart of Satisfaction Ratings')
plt.show()