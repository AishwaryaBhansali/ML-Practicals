# Implement Biased and Unbiased Multiclass classification

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score 
from sklearn.model_selection import train_test_split

df = pd.read_csv('iris.csv')
print(df.head())
print("Dataset : ", df['species'].value_counts())

x = df.drop('species', axis=1)
y = df['species']

x_train_b, x_test_b, y_train_b, y_test_b = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
model_b = LogisticRegression(max_iter=200)
model_b.fit(x_train_b, y_train_b)
pred_b = model_b.predict(x_test_b)
print("Confusion Matrix : ", confusion_matrix(pred_b, y_test_b))

df_imb = pd.concat([
    df[df['species'] == 'Iris-setosa'],
    df[df['species'] == 'Iris-versicolor'],
    df[df['species'] == 'Iris-virginica'].sample(10, random_state = 42)
])

x_i = df_imb.drop('species', axis=1)
y_i = df_imb['species']

x_train_ib, x_test_ib, y_train_ib, y_test_ib = train_test_split(x_i, y_i, test_size=0.3, random_state=42, stratify=y_i)
model_ib = LogisticRegression(max_iter=200)
model_ib.fit(x_train_ib, y_train_ib)
pred_ib = model_ib.predict(x_test_ib)
print("Confusion Matrix : ", confusion_matrix(pred_ib, y_test_ib))

print("Balanced")
print("Accuracy : ", accuracy_score(y_test_b, pred_b))
print("Precision : ", precision_score(y_test_b, pred_b, average='macro'))
print("Recall : ", recall_score(y_test_b, pred_b, average='macro'))
print("F1 Score : ", f1_score(y_test_b, pred_b, average='macro'))

print("Imbalanced")
print("Accuracy : ", accuracy_score(y_test_ib, pred_ib))
print("Precision : ", precision_score(y_test_ib, pred_ib, average='macro'))
print("Recall : ", recall_score(y_test_ib, pred_ib, average='macro'))
print("F1 Score : ", f1_score(y_test_ib, pred_ib, average='macro'))


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(pred_b, y_test_b), cmap='Blues', fmt='d', annot=True, ax=axes[0])
axes[0].set_title("Balanced Confusion Matrix")
sns.heatmap(confusion_matrix(pred_ib, y_test_ib), cmap='Reds', fmt='d', annot=True, ax=axes[1])
axes[1].set_title("Imbalanced Confusion Matrix")
plt.show()