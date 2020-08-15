import multiprocessing.dummy as mp
import networkx.algorithms as nxa
import numpy as np

from copy import deepcopy
from dataclasses import dataclass

from joblib import Parallel, delayed
from sklearn.base import TransformerMixin, BaseEstimator
from tqdm.notebook import tqdm

from .graph import get_centrality_algorithm_results, get_largest_connected_subgraph


@dataclass
class Topic():
    label: str
    qid: str
    desc: str
    score: float
    t_type: str

    @classmethod
    def from_node(cls, n, score, t_type):
        return Topic(n['label'], n['qid'], n['desc'],
                     score, t_type)

    def __eq__(self, other):
        if not isintance(other, Topic):
            return False
        return self.qid == other.qid

    def __hash__(self):
        return hash(self.qid)

    def __str__(self):
        return f"{self.label} ({self.qid})"


class LabelledTopicModel(BaseEstimator, TransformerMixin):
    """

    Parameters
    ----------
    topic_model: any
        Sklearn topic model to use. Must have the same number of components
        as the length of the labels that will be used for the topics.
    labels : list of :obj:`Topic`
        List with the topics to be asigned to each one of the topics
        of the model.
    num_topics_returned : int
        Number of topics to be assigned to each text.
    """

    def __init__(self, topic_model, topics, num_topics_returned=3):
        assert topic_model.n_components == len(topics)
        self.topic_model = topic_model
        self.topics = topics
        self.num_topics_returned = num_topics_returned

    def fit(self, X, y=None):
        self.topic_model.fit(X)

    def transform(self, X, *args, **kwargs):
        topic_distr = self.topic_model.transform(X)
        best_topics_idx = [np.argsort(text_topics)[::-1]
                           for text_topics in topic_distr]
        res = []
        for text_idx, text_topics in enumerate(best_topics_idx):
            text_res = []
            for topic_idx in text_topics[:self.num_topics_returned]:
                new_topic = deepcopy(self.topics[topic_idx])
                new_topic.score = topic_distr[text_idx][topic_idx]
                text_res.append(new_topic)
            res.append(text_res)
        return res


class TopicCombiner(BaseEstimator, TransformerMixin):
    """

    Parameters
    ----------
    """

    def __init__(self, max_num_topics=7, k=0.5, l=1.0):
        self.max_num_topics = max_num_topics
        self.k = k
        self.l = l

    def fit(self, X, y=None):
        return self

    def transform(self, X, *args, **kwargs):
        res = []
        for doc_topics in X:
            # remove duplicates
            doc_topics = list(set(doc_topics))
            topic_scores = [topic.score * self.l
                            if topic.t_type =='ner'
                            else topic.score * self.k
                            for topic in doc_topics]
            best_topics_idx = np.argsort(topic_scores)[::-1][:self.max_num_topics]
            res.append([(doc_topics[idx].label, topic_scores[idx])
                        for idx in best_topics_idx])
        return res


class TopicLabeller(BaseEstimator, TransformerMixin):
    """

    Parameters
    ----------
    graph_builder : :obj:`GraphBuilder`
        GraphBuilder instance used to build the neighbourhood graph
        of each seed term.
    r : callable
        Function used to select the node that best represents the
        contents of the generated graph.
    num_labels_per_topic : int
        Number of nodes to return as labels from the graph.
    stop_uris : list of str
        List of stop uris to be discarded when returning the
        final list of nodes.
    """

    def __init__(self, graph_builder,
                 r=nxa.centrality.information_centrality,
                 num_labels_per_topic=1,
                 stop_uris=None):
        self.graph_builder = graph_builder
        self.r = r
        self.num_labels = num_labels_per_topic
        self.stop_uris = [] if stop_uris is None else stop_uris

    def fit(self, X, y=None):
        return self

    def transform(self, X, *args, **kwargs):
        pool = mp.Pool()
        results = pool.map(self.get_topic_labels, X)
        return results

    def get_topic_labels(self, linked_entities):
        topic_neighbourhood = self.graph_builder.build_graph(linked_entities)
        subgraph = get_largest_connected_subgraph(topic_neighbourhood)
        best_nodes = get_centrality_algorithm_results(subgraph, self.r,
            self.stop_uris, self.num_labels)
        return [Topic.from_node(n, score, "ner") for n, score in best_nodes]
