# 5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
values=np.random.rand(100)
print("\n values") 
print(values)
labels=[]
for i in values[:50]:
    if i<=0.5:
        labels.append('class1')
    else:
        labels.append('class2')
labels +=[None]*50
print("\nlabels:")
print(labels)
data={
      "point":[f"x{i+1}" for i in range(100)],
      "value":values,
      "label":labels
      }
print("\n Data Frame:")
df=pd.DataFrame(data)
print(df)
num_col=df.select_dtypes(include=['int','float']).columns
for col in num_col:
    df[col].hist(figsize=(12,8),bins=10,edgecolor='black')
    plt.title(f"Histogram for {col}",fontsize=16)
    plt.xlabel(col)
    plt.ylabel("frequency")
    plt.show()
labeled_df=df[df["label"].notna()]
X_train=labeled_df[["value"]]
y_train=labeled_df["label"] 
unlabeled_df=df[df["label"].isna()].copy()
X_test=unlabeled_df[["value"]]
true_labels=["class1 "if x<=0.5 else "class2" for x in values[50:]]
k_values=[1,2,3,4,5,20,30]
results={}
accuracies={}                              
for k in k_values:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    predictions=knn.predict(X_test)
    results[k]=predictions
    accuracy=accuracy_score(true_labels,predictions)*100
    accuracies[k]=accuracy
    print(f"Accuracy for k={k}:{accuracy:.2f}%")
    unlabeled_df[f"label_k{k}"]=predictions
df1=unlabeled_df.drop(columns=['label'],axis=1)
print("\n DataFrame without 'Label':")
print(df1)