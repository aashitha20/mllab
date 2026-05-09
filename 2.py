#2

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('California Housing.csv')

df=df.select_dtypes(include=['number'])

correlation_matrix=df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True,cmap='coolwarm',
            fmt='.2f',linewidth=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

sns.pairplot(df)
plt.suptitle('Pairwise relationship between features', y=1.02)
plt.show()