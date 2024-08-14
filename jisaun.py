import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

# 读取CSV文件
results_file = 'results.csv'
results_df = pd.read_csv(results_file)

# 计算不同阈值下的precision和recall
thresholds = range(0, 101)  # 从0到100
precisions = []
recalls = []

for threshold in thresholds:
    results_df['Binary Prediction'] = results_df['Similarity'].apply(lambda x: 1 if x >= threshold else 0)
    precision = precision_score(results_df['Ground Truth'], results_df['Binary Prediction'])
    recall = recall_score(results_df['Ground Truth'], results_df['Binary Prediction'])
    precisions.append(precision)
    recalls.append(recall)

# 绘制Precision-Threshold图
plt.figure(figsize=(8, 6))  # 设置图表大小
plt.plot(thresholds, precisions, label='Precision', color='b')
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision vs. Threshold', fontsize=14)
plt.grid(True)

# 设置坐标轴样式
ax = plt.gca()
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# 设置x轴刻度间隔和y轴范围，确保原点是(0,0)
plt.xticks(np.arange(0, 101, 10))
plt.xlim(0, 100)
plt.ylim(0, 1.1)  # 将y轴范围设置到1.1
plt.tick_params(axis='y', labelsize=12)  # 增大y轴刻度字体
plt.legend()
plt.tight_layout()
plt.show()

# 绘制Recall-Threshold图
plt.figure(figsize=(8, 6))  # 设置图表大小
plt.plot(thresholds, recalls, label='Recall', color='r')
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Recall', fontsize=12)
plt.title('Recall vs. Threshold', fontsize=14)
plt.grid(True)

# 设置坐标轴样式
ax = plt.gca()
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# 设置x轴刻度间隔和y轴范围，确保原点是(0,0)
plt.xticks(np.arange(0, 101, 10))
plt.xlim(0, 100)
plt.ylim(0, 1.1)  # 将y轴范围设置到1.1
plt.tick_params(axis='y', labelsize=12)  # 增大y轴刻度字体
plt.legend()
plt.tight_layout()
plt.show()
