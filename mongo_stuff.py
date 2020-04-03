import os
from collections import namedtuple, OrderedDict
from dataclasses import dataclass, asdict, replace
from itertools import chain
from typing import List, Tuple, Dict, NamedTuple, Callable

from pymongo import DESCENDING, MongoClient
from pymongo.collection import Collection, ReturnDocument
from pymongo.database import Database

import numpy as np

from lib import color_print_warning, color_print_okblue, stub
import matplotlib.cm as cm



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


def experiment_label(coll: Collection, exp, new_label:str = None):
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


@dataclass
class Cached_Data_Descriptor:
    TAG: str = "STUFF"
    num_seeds: int = 0
    sim_time: int = 0
    tick: int = 0


def preprocess_dataset(collection: Collection, junk: Collection, exp: dict, reload_results="AUTO",
                       fields_node:Dict[str, str]=None,
                       fields_interface:Dict[str, str]= None,
                       node_breakdown_param:str = 'subtype', )->Cached_Data_Descriptor:

    def check_keys(a,b):
        int = set(a.keys()).intersection(b.keys())
        if int:
            print("Can not use same keys for record ID and field values, please rename keys:")
            print(int)
            raise KeyError()

    cache_descr = Cached_Data_Descriptor()
    ensure_indices(collection)

    trials = collection.find({"link": exp['_id']}).sort("params", DESCENDING)
    total_trials = 0
    for t in trials:
        total_trials += 1
        print("{params}:{seed}".format(**t))

    if junk.find_one({"exp_ID": exp['_id']}) is None:
        if reload_results is True or reload_results == "AUTO":
            RELOAD_RESULTS = True
        else:
            RELOAD_RESULTS = False
            print("Latest experiment does not match cached values!!! Maybe reload?")
    elif reload_results == "AUTO":
        RELOAD_RESULTS = False
    else:
        RELOAD_RESULTS = reload_results

    if RELOAD_RESULTS:
        junk.delete_many({})

        print('Grouping within MC trials')
        MC_groups = collection.aggregate([{"$match": {"link": exp['_id']}},
                                          {"$group": {"_id": "$params", "items": {"$push": "$_id"}}}], batchSize=10)

        print('Beginning parse')

        for tnum, group_data in enumerate(MC_groups):
            # extract params for convenience
            print(f"Parsing MC group {tnum}/{total_trials}: {group_data['_id']}")
            params = group_data['_id']
            if cache_descr.num_seeds == 0:
                cache_descr.num_seeds = len(group_data['items'])

            # for each trial in MC group
            for trial_id in group_data['items']:
                # get the trial itself
                trial = collection.find_one({"_id": trial_id})
                print(f"Parsing trial {trial}")
                # Get value from the config
                config = collection.find_one({"type": "CONFIG", "link": trial_id}, {'SLS.TICK': True})
                if cache_descr.tick == 0:
                    cache_descr.tick = config['SLS']['TICK']

                gl_stats = collection.find_one({"type": "SYS", "link": trial_id}, {'sim_time': True})
                if cache_descr.sim_time == 0:
                    cache_descr.sim_time = gl_stats['sim_time'] * cache_descr.tick

                node_types = collection.distinct(node_breakdown_param, {"type": "NODE", "link": trial_id})
                for node_type in node_types:
                    nodes = list(collection.find({"type": "NODE", "link": trial_id, node_breakdown_param: node_type},
                                                 list(fields_node.keys())))
                    # Make record to hold values related to this node type in this trial

                    d = {**{node_breakdown_param: node_type}, **params}
                    rec_id = junk.find_one_and_update(filter=d, update={"$set": d}, projection={"_id": 1}, upsert=True,
                                                      return_document=ReturnDocument.AFTER)["_id"]
                    # add data from nodes
                    for node in nodes:
                        updates = {v: node[k] for k, v in fields_node.items()}
                        check_keys(updates, d)

                        junk.update_one(filter={"_id": rec_id}, update={"$push": updates})
                        ifaces = list(collection.find({"type": "INTERFACE", "link": node['_id']},
                                                      list(fields_interface.keys())))
                        assert len(ifaces) == 1
                        for iface in ifaces:
                            updates = {v: iface[k] for k, v in fields_interface.items()}
                            check_keys(d, updates)
                            junk.update_one(filter={"_id": rec_id}, update={"$push": updates})

        print(f"Inserting global parameters {asdict(cache_descr)}")
        junk.insert_one(asdict(cache_descr))
        junk.insert_one({"exp_ID": exp['_id']})
    else:
        print("Loading cached values")
        st = junk.find_one({"TAG": "STUFF"})
        st = {k: st[k] for k in asdict(cache_descr).keys()}
        cache_descr = replace(cache_descr, **st)

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
            "_id":   {f"{n}": f"${n}" for n in chain(group_params, [sweep])},
            "items": {"$push": f"${field}"}
        }
    }

    sort1 = {"$sort": {f"_id.{sweep}": sort_dir}}

    group2 = {
        "$group": {
            "_id":  {f"{n}": f"$_id.{n}" for n in group_params},
            "DATA": {"$push": {f"{sweep}": f"$_id.{sweep}", "FIELD": "$items"}}
        }
    }

    sort2 = {"$sort": {f"_id.{n}": 1 for n in group_params}}
    pipeline = [match, group1, sort1, group2, sort2]

    if not quiet:
        color_print_okblue("Will run aggregate:[" + ',\n'.join([str(i) for i in pipeline])+"]")

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
