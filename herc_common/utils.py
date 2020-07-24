import functools

import dill as pickle
import numpy as np


WIKIDATA_BASE = "https://www.wikidata.org/w"

def empty_if_keyerror(function):
    """
    A decorator that wraps the passed in function and
    returns an empty string if a key error is raised.
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except KeyError:
            return ""
    return wrapper


def get_topic_terms_by_relevance(model, vectorizer, dtm_tf, top_n, lambda_):
    """ Get the term distribution of a topic based on relevance.

    This method uses the relevance formula used by pyLDAvis to customize
    the relevance of the terms that will be returned.

    Parameters
    ----------
    model
        Sklearn topic modelling algorithm with a components_ field.
    vectorizer
        Sklearn vectorizer already trained.
    dtm_tf 
        Document term matrix of the initial training corpus returned by the vectorizer.
    top_n : int
        Number of top words to be returned for each topic.
    lambda_ : float
        Float in the range [0, 1] that will be used to compute the relevance of each
        term. A value equal to 1 will return the default terms assigned to each topic,
        while values closer to 0 will return terms which are more specific.
    
    Returns
    -------
    list of list of str
        2D array where the first dimension corresponds to each topic of the model, and
        the second one to the top n terms retrieved for each topic.
    """
    vocab = vectorizer.get_feature_names()
    term_freqs = dtm_tf.sum(axis=0).getA1()
    topic_term_dists = model.components_ / model.components_.sum(axis=1)[:, None]
    term_proportion = term_freqs / term_freqs.sum()
    log_ttd = np.log(topic_term_dists)
    log_lift = np.log(topic_term_dists / term_proportion)
    relevance = lambda_ * log_ttd + (1 - lambda_) * log_lift
    return [[vectorizer.get_feature_names()[i] for i in topic.argsort()[:-top_n - 1:-1]]
            for topic in relevance]


def load_object(output_path):
    """
    """
    # see https://stackoverflow.com/questions/42960637/python-3-5-dill-pickling-unpickling-on-different-servers-keyerror-classtype
    pickle._dill._reverse_typemap['ClassType'] = type
    with open(output_path, 'rb') as file:
        res = pickle.load(file)
    return res


def save_object(obj, output_path):
    """
    """
    with open(output_path, 'wb') as file:
        pickle.dump(obj, file)