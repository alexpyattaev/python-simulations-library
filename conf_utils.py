"""Contains utility objects necessary to process configuration values and experiment setups in a sane matter

Uses plain text files or Mongodb v3 for actual storage"""
import datetime
import itertools
from functools import reduce
from operator import mul
from typing import Iterable

from pymongo.collection import Collection

__encoding__ = "utf-8"
__author__ = 'Alex Pyattaev'


class config_container(object):
    """This class should be used as container in global configs (like SLS.py)"""
    pass


def unwrap_inner(params):
    for K, V in list(params.items()):
        if not isinstance(K, tuple):
            continue
        params.pop(K)
        params[K] = []
        for vset in V:
            vset = list(vset)
            for i, v in enumerate(vset):
                if isinstance(v, str) or not isinstance(v, Iterable):
                    vset[i] = (v,)
            for pset in itertools.product(*vset):
                pset = list(pset)

                params[K].append(pset)


def unwrap_tuples(params):
    for k, v in list(params.items()):
        if isinstance(k, tuple):
            params.pop(k)
            params.update({kk: vv for kk, vv in zip(k, v)})


class Experiment(object):
    """Abstraction for Experiment

    Auto-fills database fields:
    {"type":"EXPERIMENT", "tag":tag, "time":current time}
    """

    def __init__(self, params: dict, seeds: list, storage: Collection = "point_{idx}", tag: str = ""):
        """
        :param params: Parameters for trial {key:array of values}
        :param seeds: Random trial integer seed list (used for Monte-Carlo analysis)
        :param storage: the database collection to store data
        :param tag: Optional tag to locate the experiment in DB
        """
        self.params = params
        self.tag = tag
        self.seeds = seeds
        self.db_id = None
        self.storage = storage
        unwrap_inner(self.params)
        if isinstance(storage, Collection):
            document = {"type": "EXPERIMENT", "tag": self.tag, "time": datetime.datetime.now()}

            res = storage.insert_one(document)
            self.db_id = res.inserted_id
        elif isinstance(storage, str):
            if not ('{idx}' in storage):
                raise ValueError(f'Storage file must be a template with {{idx}} field, found {storage}')
        else:
            raise TypeError('Storage must be a database collection or a filename prefix')

    def __iter__(self):
        """
        Experiment can and should be used as iterator

        :return: the next Trial to be used
        :rtype: Trial
        """

        sweep_params = {k: v for k, v in self.params.items() if not isinstance(v, str) and isinstance(v, Iterable)}
        fix_params = {k: v for k, v in self.params.items() if k not in sweep_params}
        sweep_keys, sweep_values = list(zip(*list(sweep_params.items())))

        if sweep_keys:
            # Keep index for text file export (Matlab)
            trial_idx = 0
            # Construct all possible combinations of parameter values
            for params in itertools.product(*sweep_values):
                # combine them back with their names
                p = dict(zip(sweep_keys, params))
                # Update the dict with fixed params
                p.update(fix_params)
                unwrap_tuples(p)
                for s in self.seeds:
                    trial_idx += 1
                    print("Spawning trial #{}, params{} seed {}".format(trial_idx, p, s))
                    # Create a trial
                    if isinstance(self.storage, Collection):
                        yield Trial(self, p, s, self.storage)
                    else:
                        yield Trial(self, p, s, self.storage.format(idx=trial_idx))

        else:
            trial_idx = 0
            for s in self.seeds:
                trial_idx += 1
                print("Spawning trial #{}, seed {}".format(trial_idx, s))
                # Create a trial
                if isinstance(self.storage, Collection):
                    yield Trial(self, fix_params, s, self.storage)
                else:
                    yield Trial(self, fix_params, s, self.storage.format(idx=trial_idx))

    def __len__(self):
        def ll(v):
            if isinstance(v, str):
                return 1
            try:
                return len(v)
            except TypeError:
                return 1

        if self.params:
            return reduce(mul, [ll(v) for v in self.params.values()]) * len(self.seeds)
        else:
            return len(self.seeds)


class Trial(object):
    """Abstraction for Trial, i.e. a point within an experiment set

    Auto-fills database fields:
    {"type":"TRIAL", "tag":tag, "seed":seed, "link":experiment}
    """

    def __init__(self, experiment: Experiment, params: dict, seed: int, storage: Collection):
        """
        :param  experiment: the database ID of the experiment this trial belongs to
        :param  params: Parameters for trial {key:value}
        :param  seed: Random trial seed (used for Monte-Carlo analysis)
        :param  storage: the database collection to store data
        :return: Trial
        """
        self.params = params
        self.experiment = experiment
        self.db_id = None
        self.storage = storage
        self.seed = seed
        if isinstance(storage, Collection):
            document = {"type": "TRIAL", "params": self.params, "seed": seed, "link": experiment.db_id}
            res = storage.insert_one(document)
            self.db_id = res.inserted_id
        elif isinstance(storage, str):
            pass
        else:
            raise TypeError('Storage must be a database collection or a filename prefix')

    @property
    def storage_path(self):
        if isinstance(self.storage, Collection):
            return str(self.db_id)
        else:
            return self.storage
