import json
import logging
import requests
import time

from sklearn.base import TransformerMixin, BaseEstimator
from tqdm.notebook import tqdm

from herc_common.utils import WIKIDATA_BASE


DBPEDIA_BASE = 'http://dbpedia.org'
DBPEDIA_SPOTLIGHT_BASE = 'http://api.dbpedia-spotlight.org/en'
DBPEDIA_SPOTLIGHT_MAX_CHARS = 15000
OWL_SAME_AS = 'http://www.w3.org/2002/07/owl#sameAs'

logger = logging.getLogger(__name__)

def _convert_to_wd(dbpedia_linked_entities):
    """ Links a single entity to Wikidata.

    Parameters
    ----------
    entity_label : str
        Name of the entity to be linked.
    
    Returns
    -------
    (str, str)
        Tuple where the first element is the name of the entity,
        and the second one is its 'QID' from Wikidata after linking.
    """
    wd_entities = []
    for name, url in dbpedia_linked_entities:
        resource_url = url.replace(f"{DBPEDIA_BASE}/resource", f"{DBPEDIA_BASE}/data")
        resource_url += ".json"
        res = requests.get(resource_url)
        if res.status_code != 200:
            logger.warning(f"Error loading resource '%s'. Skipping it. ", resource_url)
            continue

        res_dict = json.loads(res.content)
        try:
            mappings = res_dict[url][OWL_SAME_AS]
        except KeyError:
            wd_entities.append((name, None))

        for mapping in mappings:
            mapping_url = mapping['value']
            if 'http://www.wikidata.org/' in mapping_url:
                wd_entities.append((name, mapping_url))
                break
    return wd_entities


class DBPediaEntityLinker(BaseEstimator, TransformerMixin):
    """

    """

    def __init__(self, confidence_threshold=0.4, throttling_time=5):
        self.confidence = confidence_threshold
        self.throttling_time = throttling_time
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, *args, **kwargs):
        return [self.link_entities(text[:DBPEDIA_SPOTLIGHT_MAX_CHARS])
                for text in X]
    
    def link_entities(self, text):
        """
        """
        payload = {'confidence': self.confidence, 'text': text}
        reqheaders = {'accept': 'application/json'}
        res = requests.post(f"{DBPEDIA_SPOTLIGHT_BASE}/annotate",
                            data=payload,
                            headers={"accept": "application/json"})
        while res.status_code == 403:
            logger.warn("DBPedia spotlight limit reached. Retrying in %d seconds...",
                self.throttling_time)
            time.sleep(self.throttling_time)
            res = requests.post(f"{DBPEDIA_SPOTLIGHT_BASE}/annotate",
                            data=payload,
                            headers={"accept": "application/json"})
        
        res_dict = json.loads(res.content)
        if 'Resources' not in res_dict:
            return []
        
        return [(resource['@surfaceForm'], resource['@URI'])
                for resource in res_dict['Resources']]


class DBPedia2WikidataMapper(BaseEstimator, TransformerMixin):
    """ Links a single entity to Wikidata.

    Parameters
    ----------
    entity_label : str
        Name of the entity to be linked.
    
    Returns
    -------
    (str, str)
        Tuple where the first element is the name of the entity,
        and the second one is its 'QID' from Wikidata after linking.
    """

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, *args, **kwargs):
        return [_convert_to_wd(linked_entities)
                for linked_entities in X]


class WikidataEntityLinker(BaseEstimator, TransformerMixin):
    """ Link a list of entities to Wikidata.

    This transformer receives entities in a string form, and
    returns a tuple (entity_name, entity_uri) for each entity
    with its original name and URI in Wikidata.
    """


    def __init__(self):
        self.linked_entities_cache = {}

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, *args, **kwargs):
        return [[self.link_entity(entity) for entity in doc]
                for doc in tqdm(X)]
    
    def link_entity(self, entity_label):
        """ Links a single entity to Wikidata.

        Parameters
        ----------
        entity_label : str
            Name of the entity to be linked.
        
        Returns
        -------
        (str, str)
            Tuple where the first element is the name of the entity,
            and the second one is its 'QID' from Wikidata after linking.
        """
        if entity_label in self.linked_entities_cache:
            return (entity_label, self.linked_entities_cache[entity_label])

        url = f"{WIKIDATA_BASE}/api.php?action=wbsearchentities&search=" + \
            f"{entity_label}&language=en&format=json"
        response = requests.get(url)
        if response.status_code != 200:
            raise Error()
        
        try:
            content = json.loads(response.text)
        except:
            # invalid entity
            self.linked_entities_cache[entity_label] = None
            return self.link_entity(entity_label)

        search_results = content['search']
        if len(search_results) == 0:
            self.linked_entities_cache[entity_label] = None
            return self.link_entity(entity_label)
        
        self.linked_entities_cache[entity_label] = search_results[0]['concepturi']
        return self.link_entity(entity_label)

