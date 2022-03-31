from promptsource.templates import DatasetTemplates
from datasets import load_dataset
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score


dataset_name, subset_name = "tweet_eval","sentiment"
test_ds = load_dataset('csv', data_files='/home/jon/Documentos/ProjectoDeep/twitter_validation.csv', split='train')
actual=test_ds['label']
ag_news_prompts = DatasetTemplates(f"{dataset_name}/{subset_name}")
# Get all the prompts available in PromptSource
prompt = ag_news_prompts['sentiment_task']
results=[]
# Apply the prompt to the example
for t in range(len(test_ds)):
    result = prompt.apply(test_ds[t])
    results.append(result)
# Print a dict where the key is the pair (dataset name, subset name)
# and the value is an instance of DatasetTemplates

def predict(predictions,actual=actual):
    predicted=[]
    for a_tuple in predictions:
            if a_tuple[1] == 'positive':
                predicted.append(2)
            elif a_tuple[1] == 'negative':
                predicted.append(0)
            else:
                predicted.append(1)
    print('Accuracy',accuracy_score(actual, predicted))
    print('F1 Macro',f1_score(actual, predicted,average='macro'))
    data = {'y_Actual':  actual  ,
        'y_Predicted': predicted
        }
    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    sn.heatmap(confusion_matrix, annot=True,fmt='g')
    plt.show()
predict(results)