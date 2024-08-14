import pandas as pd
from sklearn.metrics import precision_score, recall_score

# 用来计算已经存放在csv文件夹的特定阈值的结果精度和召回率

results_df = pd.read_csv("results.csv")

# 计算precision和recall
threshold = 99
results_df['Binary Prediction'] = results_df['Similarity'].apply(lambda x: 1 if x >= threshold else 0)

precision = precision_score(results_df['Ground Truth'], results_df['Binary Prediction'])
recall = recall_score(results_df['Ground Truth'], results_df['Binary Prediction'])

# 打印结果
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

# 展示结果
summary = results_df.groupby(['Test Character', 'Test Image']).agg({
    'Ground Truth': 'sum',
    'Binary Prediction': 'sum'
}).reset_index()

summary['Precision'] = precision
summary['Recall'] = recall

print(summary)
