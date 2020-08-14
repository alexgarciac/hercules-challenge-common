from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError


def compute_similarity_scores(topics_base, topics_pred, similarity_func):
    scores_matrix = get_scores_matrix(topics_base, topics_pred, similarity_func)
    return obtain_associations_scores(scores_matrix)

def get_scores_matrix(topics_base, topics_pred, similarity_func):
    sim_measures = []
    for topic_p in topics_pred:
        p_synset = _get_synset(topic_p)
        if p_synset is None:
            array_len = len(topics_base)
            a = np.empty(array_len)
            a[:] = np.nan
            sim_measures.append(a)
            continue

        topic_sim_measures = []
        for topic_b in topics_base:
            b_synset = _get_synset(topic_b)
            if b_synset is None:
                topic_sim_measures.append(np.nan)
                continue
            try:
                similarity = getattr(p_synset, similarity_func)(b_synset)
                topic_sim_measures.append(similarity)
            except WordNetError:
                # comparing synsets with different POS
                topic_sim_measures.append(np.nan)
                continue
        sim_measures.append(topic_sim_measures)
    return np.array(sim_measures)

def obtain_associations_scores(scores_matrix):
    scores_matrix = _remove_nan_rows(scores_matrix)
    scores_matrix = _remove_nan_cols(scores_matrix)
    n = scores_matrix.shape[0]
    m = scores_matrix.shape[1]
    if n < m:
        sim_measures = np.nanmax(scores_matrix, axis=1)
    else:
        sim_measures = np.nanmax(scores_matrix, axis=0)
    return {
        'max similarity': np.max(sim_measures),
        'min similarity': np.min(sim_measures),
        'mean similarity': np.mean(sim_measures),
        'median similarity': np.median(sim_measures)
    }

def _get_synset(word):
    try:
        word = '_'.join(word.split(' '))
        return wn.synsets(word)[0]
    except IndexError:
        return None

def _remove_nan_rows(m):
    return m[~np.isnan(m).any(axis=1)]

def _remove_nan_cols(m):
    return m[:, ~np.all(np.isnan(m), axis=0)]
