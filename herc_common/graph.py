import json
import logging
import networkx as nx
import networkx.algorithms as nxa
import pdb
import requests

from herc_common.utils import empty_if_keyerror, WIKIDATA_BASE


logger = logging.getLogger(__name__)

WIKIDATA_PROPS_EXPAND = ['P31', 'P279', 'P301', 'P361', 'P366',
                         'P527', 'P910', 'P921', 'P2578', 'P2579']


def get_centrality_algorithm_results(g, algorithm, stop_uris, top_n):
    """ Return top n nodes from a graph based on the given centrality algorithm.

    Parameters
    ----------
    g: :obj:`networkx.graph`
    algorithm: callable
    stop_uris: list of str
    top_n: int

    Returns
    -------
    list of (str, float)
        List of tuples where the first element is the label of the node and the second
        one is the score obtained for the given algorithm.
    """
    metrics = algorithm(g)
    metrics = {key: val for key, val in metrics.items()
                if key not in stop_uris}
    best_qids = sorted(metrics, key=metrics.get, reverse=True)[:top_n]
    return [(g.nodes[qid], metrics[qid]) for qid in best_qids]


def get_largest_connected_subgraph(g):
    S = [g.subgraph(c).copy() for c in nxa.components.connected_components(g)]
    return max(S, key=len)

def _build_uri(entity_id):
    return f"http://www.wikidata.org/entity/{entity_id}"

@empty_if_keyerror
def _get_aliases(entity_info, lang='en'):
    return [alias['value'] 
            for alias in entity_info['aliases'][lang]]


@empty_if_keyerror
def _get_desc(entity_info, lang='en'):
    return entity_info['descriptions'][lang]['value']

@empty_if_keyerror
def _get_labels(entity_info, lang='en'):
    return entity_info['labels'][lang]['value']


class WikidataGraphBuilder():
    """ Build a Wikidata graph from a given set of seed concepts.

    This class can be used to build a graph with Wikidata
    
    Parameters
    ----------
    max_hops: int (default=2)
        Maximum depth of the graph with respect to the seed nodes used
        to build it.

    additional_props: list of str (default=None)
        List of properties to be expanded for each node in the graph. They
        will be added to the default list of properties of the graph builder.
    """

    def __init__(self, max_hops=2, additional_props=None):
        self.max_hops = max_hops
        self.props_to_expand = WIKIDATA_PROPS_EXPAND
        if additional_props:
            self.props_to_expand += additional_props
    
    def build_graph(self, terms):
        """Build the graph for the given terms."""
        logger.info("Started building graph.")
        G = nx.Graph()
        for term in terms:
            logger.debug("Seed term: %s", term[0])
            term_uri = term[1]
            if term_uri is not None:
                term_id = term_uri.split('/')[-1]
                self._add_wd_node_info(G, term_id, None, 0)
        logger.info("Finished building graph.")
        return G
    
    def _add_wd_node_info(self, graph, term_id, prev_node, curr_hop):
        logger.debug("Visiting entity '%s' - Curr hop: %d", term_id, curr_hop)
        if curr_hop > self.max_hops or term_id in ['Q4167410', 'Q4167836']:
            return
        
        # call wikidata API for uri
        endpoint = f"{WIKIDATA_BASE}/api.php?action=wbgetentities&ids={term_id}&languages=en&format=json"
        res = requests.get(endpoint)
        if res.status_code != 200:
            logger.warning("There was an error calling endpoint for term %s: %s",
                            term_id, res.content)
            raise Error()
        
        content = json.loads(res.text)
        entity_info = content['entities'][term_id]
        if 'claims' not in entity_info:
            return
        
        if term_id not in graph.nodes:
            graph.add_node(term_id)
            #graph.nodes[term_id]['alias'] = _get_aliases(entity_info)
            graph.nodes[term_id]['qid'] = term_id
            graph.nodes[term_id]['desc'] = _get_desc(entity_info)
            graph.nodes[term_id]['label'] = _get_labels(entity_info)
            graph.nodes[term_id]['n'] = curr_hop

        if prev_node is not None and not graph.has_edge(prev_node, term_id):
            graph.add_edge(prev_node, term_id)


        for claim_key, claim_values in entity_info['claims'].items():
            if claim_key not in self.props_to_expand:
                continue
            
            for value in claim_values:
                snaktype = value['mainsnak']['snaktype']
                if snaktype in ['novalue', 'somevalue']:
                    continue
                
                new_node_id = value['mainsnak']['datavalue']['value']['id']
                self._add_wd_node_info(graph, new_node_id, term_id, curr_hop + 1)
