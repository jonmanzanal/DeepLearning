from . import NLITopicClassifierWithMappingHead

BABELDOMAINS_TOPICS = [
    "0",
    "1",
    "2",
]

BABELDOMAINS_TOPIC_MAPPING = {
    "0": "0",
    "1": "1",
    "2": "2",
}


class BabelDomainsClassifier(NLITopicClassifierWithMappingHead):
    """BabelDomainsClassifier

    Specific class for topic classification using BabelDomains topic set.
    """

    def __init__(self, **kwargs):
        super(BabelDomainsClassifier, self).__init__(
            pretrained_model="roberta-large-mnli",
            labels=BABELDOMAINS_TOPICS,
            topic_mapping=BABELDOMAINS_TOPIC_MAPPING,
            query_phrase="The domain of the sentence is about",
            entailment_position=2,
            **kwargs
        )