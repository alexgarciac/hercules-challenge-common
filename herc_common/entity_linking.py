import json
import requests

from sklearn.base import TransformerMixin, BaseEstimator
from tqdm.notebook import tqdm

from herc_common.utils import WIKIDATA_BASE


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
