import hashlib
import json
import os
import time
from collections import namedtuple, OrderedDict
from dataclasses import dataclass, asdict, replace
from itertools import chain
from typing import List, Tuple, Dict, NamedTuple, Optional

import matplotlib.cm as cm
import numpy as np
from bson import ObjectId
from pymongo import DESCENDING, MongoClient
from pymongo.collection import Collection, ReturnDocument
from pymongo.database import Database
from tqdm import tqdm

from lib.code_perf_timer import Context_Timer
from lib.stuff import color_print_warning, color_print_okblue


def connect_to_results(db_server_path: str = None, client_pem="certs/client.pem",
                       server_crt='certs/ca.crt') -> Database:
    """
    Connect to typical results collection.
    If db_server_path is unspecified will load from default_db_file.txt

    :param db_server_path: The server URL to use, e.g. 'mongodb://simhost.winter.rd.tut.fi:27017'.
    :param client_pem: client certificate path
    :param server_crt: server certificate path
    :return: Database object or raises exception
    """
    if db_server_path is None:
        try:
            defdb_file = open('default_db_file.txt')
            default_db = defdb_file.readline().strip('\n')
        except IOError:
            print('Could not open default_db_file.txt')
            raise
        db_server, collection = default_db.rsplit('/', maxsplit=1)
    else:
        db_server = db_server_path
    tls_insecure = False
    if "IGNORE_TLS" in os.environ and os.environ["IGNORE_TLS"] == "TRUE":
        print("Hacker mode activated, ignoring mongodb SSL certificate errors.")
        tls_insecure = True
    client = MongoClient(host=db_server, ssl=True, ssl_ca_certs=os.path.abspath(server_crt),
                         ssl_certfile=os.path.abspath(client_pem), tlsInsecure=tls_insecure)

    authfile = open('authfile.txt')
    login, password = authfile.readline().strip('\n').split()

    print("Using credentials: {}:{}".format(login, password))
    client[login].authenticate(login, password)
    return client[login]


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
            coll.update_one(filter={'_id': exp['_id']},
                            update={'$set': {'label': new_label}})
        else:
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


def mongo_make_colors(coll, key, cmap=None):
    all_ = coll.distinct(key)
    all_.sort()
    if cmap is None:
        if len(all_) < 5:
            cmap = cm.brg
        else:
            # noinspection PyUnresolvedReferences
            cmap = cm.viridis

    COLORS = [cmap(i) for i in np.linspace(0, 1, len(all_))]

    def colors_fn(q):
        return COLORS[all_.index(q)]

    return colors_fn


def mongo_make_linestyles(coll, key, styles=('-', '--', '-.', ':')):
    all_ = coll.distinct(key)
    all_.sort()
    if len(all_) > len(styles):
        raise ValueError("Too many keys for given styleset")

    def style_fn(q):
        return styles[all_.index(q)]

    return style_fn


def find_last_experiment(collection: Collection, tag: str, only_completed: True, label: str = None) -> dict:
    """
    Find latest experiment with given tag in given collection
    :param collection: collection to search
    :param tag: experiment tag
    :param only_completed: only return experiments which successfully finished (default=True)
    :param label: only return experiments with given label (see experiment_label)
    :return: experiment object
    """

    sk = {"type": "EXPERIMENT", 'tag': tag}
    if only_completed:
        sk['time_completed'] = {"$ne": None}
    if label:
        sk['label'] = label
    exp = collection.find().sort("time", DESCENDING)[0]
    print("Experiment '{tag}':{_id} taken at {time:%d %b %Y %H:%M:%S}".format(**exp))
    return exp


@dataclass
class Cached_Data_Descriptor:
    TAG: str
    exp_ID: ObjectId
    fields_hash: str
    num_seeds: int = 0
    sim_time: int = 0
    tick: int = 0


def dump_trials(collection: Collection, exp: dict) -> int:
    trials = collection.find({"link": exp['_id']}).sort("params", DESCENDING)
    total_trials = 0
    for t in trials:
        total_trials += 1
        print("{params}:{seed}".format(**t))
    return total_trials


def preprocess_dataset(collection: Collection, junk: Collection, exp: dict, reload_results="AUTO",
                       fields_node: Dict[str, str] = None,
                       fields_components: List[Dict[str, Dict[str, str]]] = None,
                       node_breakdown_param: str = 'subtype',
                       tag: str = 'STUFF', ) -> Cached_Data_Descriptor:
    """
    Preprocess a dataset by copying only needed fileds into junk collection.
    The overall structure of the collection is essentially preserved, but the data
    from different seeds will be aggregated into lists for a given field, rather than as individual points.

    :param collection: A collection to read from (i.e. with data)
    :param junk: A collection where stuff can be written (i.e. disposable collection)
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
    :param tag: tag under which the data is dumped in the junk collection
    :returns a data descriptor which contains information about the data inserted into
            the junk collection as a result of this function

    Usage of this is only appropriate if you do not mind destroying "junk" collection. Further operations with data
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
    md5.update(all_params.encode("utf-8"))
    cache_descr = Cached_Data_Descriptor(TAG=tag, exp_ID=exp['_id'], fields_hash=md5.hexdigest())
    ensure_indices(collection)

    if junk.find_one({"exp_ID": cache_descr.exp_ID, 'fields_hash': cache_descr.fields_hash}) is None:
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
        st = junk.find_one({"TAG": tag, "exp_ID": cache_descr.exp_ID})
        st = {k: st[k] for k in asdict(cache_descr).keys()}
        cache_descr = replace(cache_descr, **st)
        return cache_descr

    # junk.delete_many({'TAG': tag})
    junk.delete_many({})

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
                    rec_id = junk.find_one_and_update(filter=d, update={"$set": d}, projection={"_id": 1}, upsert=True,
                                                      return_document=ReturnDocument.AFTER)["_id"]
                # add data from nodes
                for node in tqdm(nodes, desc=f"Nodes[{node_type}]", colour="blue", unit="node", miniters=10):
                    updates = {v: node[k] for k, v in fields_node.items()}
                    check_keys(updates, d)
                    with push_timer:
                        junk.update_one(filter={"_id": rec_id}, update={"$push": updates})
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
                            autonum = comp_desc['auto_enumerate']
                            if not autonum:
                                msg = f"""Component filter {comp_filter} produced multiple results for 
                                      node {node['_id']} this is not currently supported, and 
                                      thus you should change the filter"""
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
                                junk.update_one(filter={"_id": rec_id}, update={"$push": updates})

    print(f"Inserting global parameters {asdict(cache_descr)}")
    junk.insert_one(asdict(cache_descr))
    print(f"{find_timer.seconds=}, {upsert_timer.seconds=}, {push_timer.seconds=}")

    return cache_descr


def organize_results(coll: Collection, match_rule: dict, group_params: List[str], field: str,
                     sweep: str, sort_dir: int = 1, quiet=False) -> Dict[NamedTuple, Tuple[list, list]]:
    """
    Make a nice arrangement of data for plotting via aggregate pipeline
    :param coll: the temporary collection that holds your records
    :param match_rule: a mongodb match filter listing specific values to be found
    :param group_params: parameters used for grouping of results, e.g. if you like to make a family of plots
    :param field: the field with data values (y axis). All values will be pushed into one huge array.
    :param sweep: the field across which you want to sweep (x axis)
    :param sort_dir: sorting direction for sweep
    :param quiet: Do not print anything
    :return: Dict mapping the group_params combinations to (x_vals, y_vals)
    """

    match = {"$match": match_rule}

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

    sort2 = {"$sort": {f"_id.{n}": 1 for n in group_params}}
    pipeline = [match, group1, sort1, group2, sort2]

    if not quiet:
        color_print_okblue("Will run aggregate:[" + ',\n'.join([str(i) for i in pipeline]) + "]")

    records = coll.aggregate(pipeline)
    records = list(records)
    if not records:
        raise KeyError("No matching data found, check your match rules")
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
            y_vals.append(np.concatenate(l["FIELD"]))
        else:
            if not quiet:
                print(f"x:{x_vals}, y:{y_vals}")
            curves[key] = (x_vals, y_vals)
    return curves
