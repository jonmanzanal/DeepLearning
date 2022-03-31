
from os import access
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
test_ds = load_dataset('csv', data_files='/home/jon/Documentos/ProjectoDeep/twitter_validation.csv', split='train')
df = pd.read_csv('/home/jon/Documentos/ProjectoDeep/twitter_validation.csv')
value_count=df['label'].value_counts()
print(value_count)
ax=sns.countplot(x ='label', data = df)
ax.bar_label(ax.containers[0])
plt.show()
