import numpy as np

from tmtoolkit.topicmod.evaluate import metric_coherence_gensim


def base_scoring_function(vectorizer, texts, model, X, top_n=10, measure='u_mass'):
    return metric_coherence_gensim(measure=measure, dtm=X, 
                                   topic_word_distrib=model.components_,
                                   vocab=np.array([x for x in vectorizer.vocabulary_.keys()]), 
                                   texts=texts, return_mean=True, top_n=top_n)

def compute_model_results(model_cls, X, scoring_func, min_topics=7,
                          max_topics=30, seed=42, **kwargs):
    res = {}
    for num_topics in range(min_topics, max_topics):
        model = model_cls(n_components=num_topics, random_state=seed, **kwargs)
        model.fit(X)
        score = scoring_func(model, X)
        res[model] = score
    return res

def get_best_model(model_results):
    return max(model_results, key=model_results.get)

def print_results_info(model_results):
    best_model = get_best_model(model_results)
    print(f"Best model parameters: {best_model.get_params()}")
    print(f"Topic coherence: {model_results[best_model]}")

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
