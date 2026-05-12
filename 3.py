# 3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

file_path="iris_data.csv"
df=pd.read_csv(file_path)
features=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
scaler=StandardScaler()
df_scaled=scaler.fit_transform(df[features])

cov_matrix=np.cov(df_scaled.T)
print(cov_matrix)
eigenvalues,eignvectors=np.linalg.eig(cov_matrix)
print("Eignvalues:",eigenvalues)
print("Eignvectors:",eignvectors)

pca=PCA(n_components=2)
principal_components=pca.fit_transform(df_scaled)
df_pca=pd.DataFrame(principal_components,columns=['PC1','PC2'])
df_pca['Species']=df['Species']

plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1',y='PC2',hue='Species',data=df_pca,palette='Set1',s=100,marker='o')

plt.title('PCA of Iris Dataset (4 fratures to 2)')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.legend(title='Species')
plt.show()


# output: 

# [[ 1.00671141 -0.11010327  0.87760486  0.82344326]
#  [-0.11010327  1.00671141 -0.42333835 -0.358937  ]
#  [ 0.87760486 -0.42333835  1.00671141  0.96921855]
#  [ 0.82344326 -0.358937    0.96921855  1.00671141]]
# Eignvalues: [2.93035378 0.92740362 0.14834223 0.02074601]
# Eignvectors: [[ 0.52237162 -0.37231836 -0.72101681  0.26199559]
#  [-0.26335492 -0.92555649  0.24203288 -0.12413481]
#  [ 0.58125401 -0.02109478  0.14089226 -0.80115427]
#  [ 0.56561105 -0.06541577  0.6338014   0.52354627]]
