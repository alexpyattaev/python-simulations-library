import hashlib
import json
import os
import time
from collections import namedtuple, OrderedDict
from dataclasses import dataclass, asdict, replace
from enum import Enum
from itertools import chain
from typing import List, Tuple, Dict, NamedTuple, Optional, MutableMapping, Mapping, Set, Sequence

import numpy as np
import pytest
from bson import ObjectId
from pymongo import DESCENDING, MongoClient
from pymongo.collection import Collection, ReturnDocument
from pymongo.database import Database
from tqdm import tqdm

from lib.JSON_typing_annotations import JSONObject
from lib.code_perf_timer import Context_Timer
from lib.objwalk import string_types
from lib.stuff import color_print_warning, color_print_okblue
from bson import Int64


class Safe_Formats(Enum):
    BSON = "BSON"
    JSON = "JSON"


def make_value_safe(val, fmt: Safe_Formats = Safe_Formats.BSON) -> object:
    """Makes value safe for use in BSON (mongo) or JSON"""
    if fmt == Safe_Formats.BSON:
        if isinstance(val, np.generic):
            val = val.item()

        if isinstance(val, int):
            if abs(val) > 2 ** 31:
                return Int64(val)
            else:
                return int(val)
        return val
    elif fmt == Safe_Formats.JSON:
        if isinstance(val, np.generic):
            return val.item()
        return val


def make_data_safe(kw: JSONObject, fmt: Safe_Formats = Safe_Formats.BSON) -> JSONObject:
    """Makes container (dict,list,set) safe for use in BSON (mongo) or JSON, operates in place where possible"""

    for k, v in list(kw.items()):
        # dive into mapping data types
        if isinstance(v, MutableMapping):
            # noinspection PyTypeChecker
            make_data_safe(v, fmt)
            continue
        elif isinstance(v, Mapping):
            # force immutable mappings to become mutable
            kw[k] = make_data_safe(dict(v), fmt)
            continue
        # destructure numpy arrays as lists
        if isinstance(v, np.ndarray):
            v = v.tolist()

        # convert sequences into native types (unless they are strings)
        if isinstance(v, (Sequence, Set)) and not isinstance(v, string_types):
            typ = type(v)
            kw[k] = typ([make_value_safe(i, fmt) for i in v])
        else:
            kw[k] = make_value_safe(v, fmt)

    return kw


@pytest.fixture
def unsafe_document():
    return {"a": np.full([3, 3], 5, dtype=float),
            "b": "some string", "c": np.zeros(1, dtype=np.int64)[0], "d": 2 ** 54}


def test_make_json_safe(unsafe_document):
    make_data_safe(unsafe_document, fmt=Safe_Formats.JSON)
    assert len(unsafe_document['a']) == 3, "Numpy array should have been serialized"
    assert type(unsafe_document['c']) == int, "JSON does not support numpy int"
    assert unsafe_document['b'] == "some string", "Strings should not get touched"


def test_make_bson_safe(unsafe_document):
    make_data_safe(unsafe_document, fmt=Safe_Formats.BSON)
    assert len(unsafe_document['a']) == 3, "Numpy array should have been serialized"
    assert unsafe_document['b'] == "some string", "Strings should not get touched"
    assert type(unsafe_document['d']) == Int64, "BSON does not support bigint"


class ConnectionBuilder:
    _db_server: Optional[str] = None
    _db_collection: Optional[str] = None
    _client_pem: str = "certs/client.pem"
    _server_crt: str = 'certs/ca.crt'
    _tls_insecure: bool = False
    _server_url_configured = False
    _login_configured = False
    _login = None
    _password = None

    def __init__(self):
        if "IGNORE_TLS" in os.environ and os.environ["IGNORE_TLS"] == "TRUE":
            print("Hacker mode activated, ignoring mongodb SSL certificate errors.")
            self._tls_insecure = True

    def with_silent_defaults(self):
        """
        Use defaults for everything
        """
        self.with_default_db()
        self.with_default_login_password()
        return self

    def with_insecure_TLS(self):
        """
           Ignore TLS validation
        """
        self._tls_insecure = True
        return self

    def with_login_password(self, login: str, password: str):
        self._login = login
        self._password = password
        self._login_configured = True
        return self

    def with_default_login_password(self, authfile_name: str = "authfile.txt"):
        authfile = open(authfile_name)
        self._login, self._password = authfile.readline().strip('\n').split()
        self._login_configured = True
        return self

    def with_db_path(self, db_server_path: str):
        proto, server_url = db_server_path.split("//")
        try:
            db_server, self._db_collection = server_url.rsplit('/', maxsplit=1)
        except:
            db_server = server_url
            self._db_collection = ""

        self._db_server = proto + "//" + db_server
        self._server_url_configured = True
        return self

    def with_default_db(self):
        try:
            defdb_file = open('default_db_file.txt')
            default_db = defdb_file.readline().strip('\n')
        except IOError:
            print('Could not open default_db_file.txt')
            raise
        self._db_server, self._db_collection = default_db.rsplit('/', maxsplit=1)
        self._server_url_configured = True
        return self

    def with_custom_TLS(self, client_pem: str, server_crt: str):
        self._client_pem = client_pem
        self._server_crt = server_crt
        return self

    def connect(self) -> Database:
        assert self._server_url_configured, "URL must be configured"
        assert self._login_configured, "Login must be configured"
        print(f"Connecting to mongodb server {self._db_server}")
        print(f"Using credentials: {self._login}:{self._password}, collection '{self._db_collection}'")

        client = MongoClient(host=self._db_server, ssl=True,
                             tlsCAFile=os.path.abspath(self._server_crt),
                             authSource=self._login,
                             username=self._login, password=self._password,
                             tlsCertificateKeyFile=os.path.abspath(self._client_pem), tlsInsecure=self._tls_insecure)

        return client[self._login] if not self._db_collection else client[self._login][self._db_collection]


def ensure_indices(collection: Collection, drop_current=False, index_base_name="SLS_experiment_idx"):
    """
    Ensures normal indexes are created on collection for result lookup
    :param drop_current: drop current indices (if any)
    :param index_base_name: base name for new indices to be created
    :param collection: collection to work on
    :return:
    """
    if drop_current:
        info = collection.index_information()
        for idx in info.keys():
            if not idx.startswith("_"):
                collection.drop_index(idx)

    idx_list = [[('type', DESCENDING), ('link', DESCENDING)], [('link', DESCENDING)]]
    for idx, keys, in enumerate(idx_list):
        name = f"{index_base_name}_{idx}"
        if name not in collection.index_information():
            collection.create_index(keys, name=name, unique=False, sparse=True)


def experiment_label(coll: Collection, exp: dict, new_label: str = None) -> Optional[str]:
    """
    Set or remove experiment label
    :param coll: collection to use
    :param exp: experiment document
    :param new_label: label to set, empty string will erase label. None returns the existing label
    :return: updated experiment label
    """
    if new_label is not None:
        if new_label != "":
            print("Adding label {new_label} to experiment '{tag}':{_id} taken at {time:%d %b %Y %H:%M:%S}".format(
                new_label=new_label, **exp))
            coll.update_one(filter={'_id': exp['_id']},
                            update={'$set': {'label': new_label}})
        else:
            print("Popping label from experiment '{tag}':{_id} taken at {time:%d %b %Y %H:%M:%S}".format(**exp))
            coll.update_one(filter={'_id': exp['_id']},
                            update={'$unset': 'label'})
    try:
        return coll.find_one(filter={'_id': exp['_id']}, projection=['label'])['label']
    except KeyError:
        return None


def coord(x) -> NamedTuple:
    return namedtuple('C', sorted(x))(**x)


def recursive_clean(collection: Collection, obj, pointer_name='link') -> int:
    """
    Clears recursively everything that links to obj.
    :param collection: Collection to operate on
    :param obj: "root" object
    :param pointer_name: name of the linking field (do not change unless you know what you are doing=)
    :return: number of objects deleted
    """
    oid = obj['_id']
    to_clean = collection.find({pointer_name: oid}, {'_id': 1})
    c = collection.delete_many({'_id': oid}).deleted_count
    for o in to_clean:
        c += recursive_clean(collection, o)
    return c


def mongo_make_colors(coll, key: str, categories: str = None, cmap=None):
    import matplotlib.cm as cm

    if cmap is None:
        cmap = cm.viridis

    COLORS = {}

    all_keys = coll.distinct(key)
    all_keys.sort()
    if categories is not None:

        all_categories = coll.distinct(categories)
        all_categories.sort()
        ranges1 = np.linspace(0, 1, len(all_categories), endpoint=False)
        ranges2 = np.linspace(ranges1[1], 1, len(all_categories), endpoint=True) - ranges1[1] / len(all_keys)

        for i, cat in enumerate(all_categories):
            for j, v in enumerate(np.linspace(ranges1[i], ranges2[i], len(all_keys), endpoint=True)):
                COLORS[(cat, all_keys[j])] = cmap(v)
    else:
        for j, v in enumerate(np.linspace(0, 1, len(all_keys))):
            COLORS[(all_keys[j],)] = cmap(v)

    def colors_fn(*q):
        return COLORS[q]

    return colors_fn


LINE_STYLES_COLLECTION = {
    'solid': (0, ()),
    'dotted': (0, (1, 1)),
    'dashed': (0, (5, 1)),
    'dashdot': 'dashdot',
    'long dash with offset': (5, (10, 3)),
    'densely dashdotted': (0, (3, 1, 1, 1)),
    'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1)),
    'dashdotdotted': (0, (3, 2, 1, 2, 1, 2)),
    'loosely dashdotted': (0, (3, 10, 1, 10)),
    'loosely dotted': (0, (1, 5)),
    'dashdotted': (0, (3, 5, 1, 5)),
    'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
}


def test_linestyles():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    X, Y = np.linspace(0, 100, 10), np.zeros(10)
    yticklabels = []

    for i, (name, linestyle) in enumerate(LINE_STYLES_COLLECTION.items()):
        ax.plot(X, Y + i, linestyle=linestyle, linewidth=1.5, color='black')
        yticklabels.append(name)

    ax.set_title("line styles")
    ax.set(ylim=(-0.5, len(LINE_STYLES_COLLECTION) - 0.5),
           yticks=np.arange(len(LINE_STYLES_COLLECTION)),
           yticklabels=yticklabels)
    ax.tick_params(left=False, bottom=False, labelbottom=False)
    ax.spines[:].set_visible(False)

    # For each line style, add a text annotation with a small offset from
    # the reference point (0 in Axes coords, y tick value in Data coords).
    for i, (name, linestyle) in enumerate(LINE_STYLES_COLLECTION.items()):
        ax.annotate(repr(linestyle),
                    xy=(0.0, i), xycoords=ax.get_yaxis_transform(),
                    xytext=(-6, -12), textcoords='offset points',
                    color="blue", fontsize=8, ha="right", family="monospace")

    plt.tight_layout()
    plt.show()


def mongo_make_linestyles(coll, key, styles=tuple(LINE_STYLES_COLLECTION.values())):
    all_ = coll.distinct(key)
    all_.sort()
    if len(all_) > len(styles):
        raise ValueError("Too many keys for given styleset")

    def style_fn(q):
        return styles[all_.index(q)]

    return style_fn


def find_experiments(collection: Collection, tag: str, only_completed: True, label: str = None, quiet=False) -> List[
    dict]:
    """
        Find all experiments with given tag in given collection
        :param collection: collection to search
        :param tag: experiment tag
        :param only_completed: only return experiments which successfully finished (default=True)
        :param label: only return experiments with given label (see experiment_label)
        :param quiet: do not print info messages
        :return: experiment object
        """
    sk = {"type": "EXPERIMENT", 'tag': tag}
    if only_completed:
        sk['time_completed'] = {"$exists": True, "$ne": None}
    if label:
        sk['label'] = label
    if not quiet:
        print("Search key is ", sk)
    exps = list(collection.find(sk).sort("time", DESCENDING))
    if not exps and not quiet:
        print(f"Could not find anything for search key {sk}")
    return exps


def make_watermark(exp: dict) -> str:
    """Makes a watermark text for a given experiment
    :param exp: Experiment dict
    """
    return "{tag} {time_completed:%d.%m.%y %H:%M} {_id}".format(**exp)


def find_last_experiment(collection: Collection, tag: str, only_completed: True, label: str = None, quiet=False) -> \
        Optional[dict]:
    """
    Find latest experiment with given tag in given collection
    :param collection: collection to search
    :param tag: experiment tag
    :param only_completed: only return experiments which successfully finished (default=True)
    :param label: only return experiments with given label (see experiment_label)
    :return: experiment object
    """

    try:
        exp = find_experiments(collection, tag, only_completed, label, quiet)[0]
    except IndexError:
        return None
    if not quiet:
        print("Experiment '{tag}':{_id} taken at {time:%d %b %Y %H:%M:%S}".format(**exp))
    return exp


@dataclass
class Cached_Data_Descriptor:
    TAG: str
    exp_ID: ObjectId
    fields_hash: str
    all_fields: str = ""
    num_seeds: int = 0
    sim_time: float = 0
    tick: int = 0


def dump_trials(collection: Collection, exp: dict) -> int:
    trials = collection.find({"link": exp['_id']}).sort("params", DESCENDING)
    total_trials = 0
    for t in trials:
        total_trials += 1
        print("{params}:{seed}".format(**t))
    return total_trials


def check_cached_scratch_data(scratch: Collection, cache_descr: Cached_Data_Descriptor,
                              reload_results: str = "AUTO") -> Tuple[bool, Cached_Data_Descriptor]:
    if reload_results is True:
        print("Forced reloading of cached data...")
        return True, cache_descr

    cached = scratch.find_one({"exp_ID": cache_descr.exp_ID, 'fields_hash': cache_descr.fields_hash})
    if cached is None:
        if reload_results == "AUTO":
            print("No suitable cached data, recalculating...")
            return True, cache_descr
        else:
            print("Latest experiment does not match cached values!!! Maybe reload?")
            return False, cache_descr

    else:
        print("Loading cached values")
        st = {k: cached.get(k, None) for k in asdict(cache_descr).keys()}
        cache_descr = replace(cache_descr, **st)
        return False, cache_descr





def preprocess_dataset_simple(collection: Collection, scratch: Collection, exp: dict,
                              fields: Dict[str, str] = None, extract_from: str = "RESULTS", reload_results="AUTO",
                              tag: str = "STUFF") -> Cached_Data_Descriptor:
    find_timer = Context_Timer()
    upsert_timer = Context_Timer()
    push_timer = Context_Timer()

    ensure_indices(collection)

    md5 = hashlib.md5()
    all_params = json.dumps(fields, sort_keys=True) + tag

    # print("="*40)
    # print(all_params)
    md5.update(all_params.encode("ASCII"))
    cache_descr = Cached_Data_Descriptor(TAG=tag, exp_ID=exp['_id'], all_fields=all_params, fields_hash=md5.hexdigest())
    print(cache_descr)
    need_reload, cache_descr = check_cached_scratch_data(scratch=scratch, cache_descr=cache_descr, reload_results=reload_results)
    if not need_reload:
        return cache_descr

    scratch.delete_many({})
    print('Grouping within MC trials')
    MC_groups = list(collection.aggregate([{"$match": {"link": exp['_id']}},
                                           {"$group": {"_id": "$params", "items": {"$push": "$_id"}}}], batchSize=10))

    print('Beginning parse')

    for group_data in tqdm(MC_groups, desc="Parameter groups", colour="red", unit="group"):
        # print(f"Parsing MC group {group_data['_id']}")
        params = group_data['_id']
        if cache_descr.num_seeds == 0:
            cache_descr.num_seeds = len(group_data['items'])

        # for each trial in MC group
        for trial_id in tqdm(group_data['items'], desc="Trials", colour="green", unit="trial"):
            # Get value from the config
            with find_timer:
                nodes = list(collection.find({"type": extract_from, "link": trial_id},
                                             list(fields.keys())))
            # Make record to hold values related to this node type in this trial

            d = params.copy()

            with upsert_timer:
                rec_id = scratch.find_one_and_update(filter=d, update={"$set": d}, projection={"_id": 1}, upsert=True,
                                                     return_document=ReturnDocument.AFTER)["_id"]
            # add data from nodes
            for node in tqdm(nodes, desc=f"Going over {extract_from} records...", colour="blue", unit="node", miniters=10):
                updates = {v: node[k] for k, v in fields.items()}

                with push_timer:
                    scratch.update_one(filter={"_id": rec_id}, update={"$push": updates})

    print(f"Inserting global parameters {asdict(cache_descr)}")
    scratch.insert_one(asdict(cache_descr))
    print(f"{find_timer.seconds=}, {upsert_timer.seconds=}, {push_timer.seconds=}")

    return cache_descr


def preprocess_dataset(collection: Collection, scratch: Collection, exp: dict, reload_results="AUTO",
                       fields_node: Dict[str, str] = None,
                       fields_components: List[Dict[str, Dict[str, str]]] = None,
                       node_breakdown_param: str = 'subtype',
                       tag: str = 'STUFF', ) -> Cached_Data_Descriptor:
    """
    Preprocess a dataset by copying only needed fileds into scratch collection.
    The overall structure of the collection is essentially preserved, but the data
    from different seeds will be aggregated into lists for a given field, rather than as individual points.

    :param collection: A collection to read from (i.e. with data)
    :param scratch: A collection where stuff can be written (i.e. disposable collection)
    :param exp: the Experiment document
    :param reload_results: set to True to always reload, False to never reload, AUTO to decide automatically

    :param fields_node: Which keys to extract from nodes and how to save them, e.g.
             fields_node = {'data_bytes_delivered': "throughput",
                            'data_bytes_generated': "generated",
                            'data_bytes_dropped': "dropped",
                            'pos': 'pos'}
    :param fields_components: Which keys to extract from e.g. interfaces and how to save them, e.g.
             fields_components = [{"filter": {"type": "INTERFACE", "subtype": "wifi"},
                     "project": {"tx_power": "tx_power", "csma_rts_attempts": "rts_attempts"}},
                    {"filter": {"type": "INTERFACE", "subtype": "lte"},
                     "project": {"tx_power": "tx_power", "cell_id": "cell_id"}},
                    {"filter": {"type": "COMPUTE"},
                     "project": {"jobs_handled": "jobs_handled"}
                     }]
    :param node_breakdown_param: additional node category breakdown field (like, is it a BS or client)
    :param tag: tag under which the data is dumped in the scratch collection
    :returns a data descriptor which contains information about the data inserted into
            the scratch collection as a result of this function

    Usage of this is only appropriate if you do not mind destroying "scratch" collection. Further operations with data
    can be made either directly or with organize_results() function.
    """

    find_timer = Context_Timer()
    upsert_timer = Context_Timer()
    push_timer = Context_Timer()

    def check_keys(a, b):
        isect = set(a.keys()).intersection(b.keys())
        if isect:
            print("Can not use same keys for record ID and field values, please rename keys:")
            print(isect)
            raise KeyError()

    md5 = hashlib.md5()
    all_params = json.dumps(fields_node, sort_keys=True) + json.dumps(fields_components, sort_keys=True)
    all_params += node_breakdown_param + tag
    # print("="*40)
    # print(all_params)
    md5.update(all_params.encode("ASCII"))
    cache_descr = Cached_Data_Descriptor(TAG=tag, exp_ID=exp['_id'], all_fields=all_params, fields_hash=md5.hexdigest())
    ensure_indices(collection)
    # print(cache_descr.fields_hash)
    # print("=" * 40)
    # exit()
    if scratch.find_one({"exp_ID": cache_descr.exp_ID, 'fields_hash': cache_descr.fields_hash}) is None:
        if reload_results is True or reload_results == "AUTO":
            RELOAD_RESULTS = True
            if reload_results == "AUTO":
                print("Found mismatch in cached data, recalculating...")
            else:
                print("Forced reloading of cached data...")
        else:
            RELOAD_RESULTS = False
            print("Latest experiment does not match cached values!!! Maybe reload?")
    elif reload_results == "AUTO":
        RELOAD_RESULTS = False
    else:
        RELOAD_RESULTS = reload_results

    if not RELOAD_RESULTS:
        print("Loading cached values")
        st = scratch.find_one({"TAG": tag, "exp_ID": cache_descr.exp_ID})
        st = {k: st.get(k, None) for k in asdict(cache_descr).keys()}
        cache_descr = replace(cache_descr, **st)
        return cache_descr

    # scratch.delete_many({'TAG': tag})
    scratch.delete_many({})

    print('Grouping within MC trials')
    MC_groups = list(collection.aggregate([{"$match": {"link": exp['_id']}},
                                           {"$group": {"_id": "$params", "items": {"$push": "$_id"}}}], batchSize=10))

    print('Beginning parse')

    for group_data in tqdm(MC_groups, desc="Parameter groups", colour="red", unit="group"):
        # print(f"Parsing MC group {group_data['_id']}")
        params = group_data['_id']
        if cache_descr.num_seeds == 0:
            cache_descr.num_seeds = len(group_data['items'])

        # for each trial in MC group
        for trial_id in tqdm(group_data['items'], desc="Trials", colour="green", unit="trial"):
            # Get value from the config

            if cache_descr.tick == 0:
                with find_timer:
                    config = collection.find_one({"type": "CONFIG", "link": trial_id}, {'SLS.TICK': True})
                cache_descr.tick = config['SLS']['TICK']

            if cache_descr.sim_time == 0:
                with find_timer:
                    gl_stats = collection.find_one({"type": "SYS", "link": trial_id}, {'sim_time': True})
                cache_descr.sim_time = gl_stats['sim_time'] * cache_descr.tick

            node_types = collection.distinct(node_breakdown_param, {"type": "NODE", "link": trial_id})
            for node_type in node_types:
                with find_timer:
                    nodes = list(collection.find({"type": "NODE", "link": trial_id, node_breakdown_param: node_type},
                                                 list(fields_node.keys())))
                # Make record to hold values related to this node type in this trial

                d = {**{node_breakdown_param: node_type}, **params}

                with upsert_timer:
                    rec_id = scratch.find_one_and_update(filter=d, update={"$set": d}, projection={"_id": 1}, upsert=True,
                                                         return_document=ReturnDocument.AFTER)["_id"]
                # add data from nodes
                for node in tqdm(nodes, desc=f"Nodes[{node_type}]", colour="blue", unit="node", miniters=10):
                    updates = {v: node[k] for k, v in fields_node.items()}
                    check_keys(updates, d)
                    with push_timer:
                        scratch.update_one(filter={"_id": rec_id}, update={"$push": updates})
                    if fields_components is None:
                        continue
                    assert isinstance(fields_components, list), "Expecting list for fields_components"
                    for comp_desc in fields_components:
                        comp_filter, comp_projection = comp_desc['filter'], comp_desc['project']
                        comp_filter["link"] = node['_id']
                        with find_timer:
                            components = list(collection.find(comp_filter, list(comp_projection.keys())))
                        # print(comp_desc)
                        if len(components) > 1:
                            autonum = comp_desc.get('auto_enumerate', False)
                            if not autonum:
                                msg = f"""Component filter {comp_filter} produced multiple results for 
                                      node {node['_id']} this is not currently supported, and 
                                      thus you should change the filter or set auto_enumerate:True"""
                                raise ValueError(msg)
                        else:
                            autonum = False

                        for i, c in enumerate(components):
                            # print(i, c)
                            suff = f"_{i}" if autonum else ""
                            try:
                                updates = {v + suff: c[k] for k, v in comp_projection.items()}
                            except KeyError as e:
                                raise KeyError(f"Field '{e}' not found in component obtained by {comp_filter}")
                            check_keys(d, updates)
                            with push_timer:
                                scratch.update_one(filter={"_id": rec_id}, update={"$push": updates})

    print(f"Inserting global parameters {asdict(cache_descr)}")
    scratch.insert_one(asdict(cache_descr))
    print(f"{find_timer.seconds=}, {upsert_timer.seconds=}, {push_timer.seconds=}")

    return cache_descr


def organize_results(coll: Collection, match_rule: dict, group_params: List[str], field: str,
                     sweep: str, sort_dir: int = 1, quiet=False) -> Dict[NamedTuple, Tuple[list, list]]:
    """
    Make a nice arrangement of data for plotting via aggregate pipeline
    :param coll: the temporary collection that holds your records
    :param match_rule: a mongodb match filter listing specific values to be found
    :param group_params: parameters used for grouping of results to make a family of plots. Empty group means one curve.
    :param field: the field with data values (y axis). All values will be pushed into one huge array.
    :param sweep: the field across which you want to sweep (x axis)
    :param sort_dir: sorting direction for sweep
    :param quiet: Do not print anything
    :return: Dict mapping the group_params combinations to (x_vals, y_vals)
    """

    matchkey = {"$match": match_rule}

    group1 = {
        "$group": {
            "_id": {f"{n}": f"${n}" for n in chain(group_params, [sweep])},
            "items": {"$push": f"${field}"}
        }
    }

    sort1 = {"$sort": {f"_id.{sweep}": sort_dir}}

    group2 = {
        "$group": {
            "_id": {f"{n}": f"$_id.{n}" for n in group_params},
            "DATA": {"$push": {f"{sweep}": f"$_id.{sweep}", "FIELD": "$items"}}
        }
    }
    pipeline = [matchkey, group1, sort1, group2]
    if group_params:
        sort2 = {"$sort": {f"_id.{n}": 1 for n in group_params}}
        pipeline.append(sort2)

    if not quiet:
        color_print_okblue("Will run aggregate:[" + ',\n'.join([str(i) for i in pipeline]) + "]")

    records = coll.aggregate(pipeline)
    records = list(records)
    if not records:
        raise KeyError(f"No data found for {matchkey}!")
    if not quiet:
        color_print_okblue(f"Got records: {records}")

    curves: OrderedDict[NamedTuple, Tuple[list, list]] = OrderedDict()
    for rec in records:
        if not quiet:
            color_print_okblue(rec["_id"])
        key = coord(rec["_id"])
        x_vals = []
        y_vals = []
        for l in rec["DATA"]:
            if not l["FIELD"]:
                if not quiet:
                    color_print_warning(f'No data points found for {rec["_id"]}')
                break
            x_vals.append(l[sweep])
            fdata = l["FIELD"]
            try:
                y_vals.append(np.concatenate(fdata))  # original data was a list
            except ValueError:
                y_vals.append(fdata)  # original data was a point
        else:
            if not quiet:
                print(f"x:{x_vals}, y:{y_vals}")
            curves[key] = (x_vals, y_vals)
    return curves


def get_progress(collection, tag):
    exp = find_last_experiment(collection, tag=tag, only_completed=False, quiet=True)

    if exp is None:
        raise FileNotFoundError()

    trials = collection.find({"link": exp['_id']}).sort("params", DESCENDING)
    done = 0
    total = 0
    failed = 0
    for t in trials:
        sys_data = collection.find_one({"type": "SYS", "link": t['_id']},
                                       ['termination_condition', 'termination_condition_code'])
        total += 1
        if sys_data is not None:
            if sys_data['termination_condition_code'] > 0:
                done += 1
            else:
                failed += 1

    return total, done, failed


def watch_progress(collection, tag, refresh_sec=5.0) -> None:
    """Watch progress dynamically
    :param collection: collection to pull data from
    :param tag: experiment tag
    :param refresh_sec: refresh rate
    :return None once the runner finishes its work and results are safe to process.
    """
    total, done, failed = get_progress(collection, tag)
    print(f"Done {done}/{total} ({done / total * 100}%)")
    marked = done + failed

    with tqdm(initial=marked, total=total, desc="Done", unit="trial", mininterval=0, miniters=1) as tq:
        try:
            while marked < total:
                while marked < (done + failed):
                    marked += 1
                    tq.update()
                    tq.set_postfix({"done": done, "failed": failed})
                    tq.refresh()
                time.sleep(refresh_sec)
                total, done, failed = get_progress(collection, tag)
        except KeyboardInterrupt:
            return


@dataclass
class Histogram_Data:
    """
    Stores histogram data after pulled from database for quick plotting. Also supports scaling bounds correctly.
    """
    bin: float
    hi: float
    lo: float
    toobig: int
    toosmall: int
    vals: np.ndarray

    @property
    def X_axis(self):
        return np.linspace(self.lo, self.hi - self.bin, len(self.vals))

    def __iadd__(self, other):
        assert isinstance(other, Histogram_Data)
        for n in ["bin", "hi", "lo"]:
            assert np.isclose(getattr(self, n), getattr(other, n)), f"Incompatible value for {n}"

        for n in ["toobig", "toosmall", "vals"]:
            setattr(self, n, getattr(self, n) + getattr(other, n))
        return self

    def __post_init__(self):
        if not isinstance(self.vals, np.ndarray):
            self.vals = np.array(self.vals)

    def scale(self, scale=1.0):
        self.lo *= scale
        self.hi *= scale
        self.bin *= scale
        return self
