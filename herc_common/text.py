import string

import matplotlib.pyplot as plt
import spacy

from collections import Counter

from sklearn.base import TransformerMixin, BaseEstimator
from spacy import displacy
from tqdm.notebook import tqdm
from wordcloud import WordCloud


def plot_word_cloud(text):
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


class NamedEntityRecognizer(BaseEstimator, TransformerMixin):
    def __init__(self, spacy_model, disable=None, min_entity_counts=None,
                 max_entities=None):
        self.nlp = spacy_model.load()
        self.disable = disable if disable is not None else []
        self.min_entity_counts = min_entity_counts
        self.max_entities = max_entities
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, *args, **kwargs):
        entities_texts = [self.get_entities(text) for text in X]
        if self.min_entity_counts is None:
            return entities_texts
        
        return [[entity_label 
                for entity_label, entity_count in Counter(entities_text).most_common(self.max_entities)
                if entity_count >= self.min_entity_counts]
                for entities_text in entities_texts]
    
    def get_entities(self, text):
        doc = self.nlp(text)
        return [x.text for x in doc.ents 
                if x.label_ not in self.disable
                and len(x.text) > 2]
    
    def get_most_common_entities(self, text, n=10):
        entities = self.get_entities(text)
        return Counter(entities).most_common(n)
    
    def visualize_entities(self, text, jupyter=True):
        doc = self.nlp(text)
        displacy.render(doc, jupyter=jupyter, style='ent')


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, spacy_model, additional_stopwords=None, disable=None):
        self.disable = disable if disable is not None else []
        self.custom_stop_words = additional_stopwords if additional_stopwords is not None else []
        self.nlp = spacy_model.load(disable=self.disable)

    def fit(self, X, y=None):
        return self

    def transform(self, X, *args, **kwargs):
        return self._preprocess_docs(X)
    
    def _preprocess_docs(self, X):
        preproc_pipe = []
        for doc in tqdm(self.nlp.pipe(X)):
            preproc_pipe.append(self._get_doc_tokens(doc))
        return preproc_pipe
    
    def _get_doc_tokens(self, doc):
        return [t.lemma_ for t in doc if len(t.text) > 2 and
                not self._is_stop_word(t) and t.text not in string.punctuation
                and t.is_alpha and not t.is_digit]

    def _is_stop_word(self, token):
        return token.is_stop or token.text.lower() in self.custom_stop_words
