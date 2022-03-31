from ast import List
from difflib import context_diff
from a2t.data import Dataset
from a2t.tasks import TopicClassificationFeatures
from a2t.base import EntailmentClassifier , SentimentClassifier
from a2t.tasks import TopicClassificationTask
import pandas as pd
from datasets import load_dataset
from datasets import Dataset as Datast
import csv
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score


labels=[0,1,2]
test_ds = load_dataset('csv', data_files='/home/jon/Documentos/ProjectoDeep/twitter_validation.csv', split='train')
actual=test_ds["label"]
text=test_ds["text"]
class TopicClassificationDataset(Dataset):
    def __init__(self) -> None:
        
        super().__init__(labels=labels)
        for i in range(len(actual)):
                context= text[i]
                label= actual[i]
                self.append(TopicClassificationFeatures(context=context, label=label))
dataset = TopicClassificationDataset()
task = TopicClassificationTask(name="Sentiment task", labels=labels)
model = SentimentClassifier(
    'cardiffnlp/twitter-roberta-base-sentiment', 
    use_tqdm=False, 
    use_cuda=True, 
    half=True
)
predictions = model(
    task=task, 
    features=dataset, 
    return_labels=True, 
    return_confidences=True, 
    topk=1
)
def predict(predictions,actual=actual):
    predicted=[]
    for a_tuple in predictions:
        predicted.append(int(a_tuple[0]))
    print('cardiffnlp/twitter-roberta-base-sentiment Accuracy',accuracy_score(actual, predicted))
    print('cardiffnlp/twitter-roberta-base-sentiment F1 Macro',f1_score(actual, predicted,average='macro'))
    data = {'y_Actual':  actual  ,
        'y_Predicted': predicted
        }
    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True,fmt='g')
    plt.show()
predict(predictions)
