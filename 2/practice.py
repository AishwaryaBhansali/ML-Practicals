import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('breast_cancer.csv')
print(df.head())

df = df.drop('id', axis=1)
print(df.head())

x = df.drop('diagnosis', axis=1)
y = df['diagnosis']

scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

colors = {'M' : 'red', 'B' : 'green'}
plt.figure(figsize=(12,5))
plt.scatter(x_pca[:,0], x_pca[:,1], c=y.map(colors), s=60, alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Breast Cancer Dataset")
plt.show()