from typing import List

from a2t.tasks.text_classification import TopicClassificationFeatures
from .base import Dataset
from datasets import load_dataset
class SentimentTopicClassificationDataset(Dataset):
    """A class to handle BabelDomains datasets.

    This class converts BabelDomains data files into a list of `TopicClassificationFeatures`.
    """

    def __init__(self, input_path: str, labels: List[str], *args, **kwargs) -> None:
        """
        Args:
            input_path (str): The path to the input file.
            labels (List[str]): The possible label set of the dataset.
        """
        super().__init__(labels=labels, *args, **kwargs)
        test_ds = load_dataset('csv', data_files=input_path, split='train')
        actual=test_ds["label"]
        text=test_ds["text"]
        for i in range(len(actual)):
                context= text[i]
                label= str(actual[i])
                self.append(TopicClassificationFeatures(context=context, label=label))