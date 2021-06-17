#!/usr/bin/env python
# Script LGPL from https://raw.githubusercontent.com/epage/PythonUtils/master/util/linux.py
# Script modified by Alex Pyattaev
import itertools
import mmap
import os
import unittest
import warnings
from hashlib import md5
from typing import Tuple, Iterable, BinaryIO

import psutil

try:
    from xdg import BaseDirectory
except ImportError:
    BaseDirectory = None

_libc = None


def get_file_hash(fileobj: BinaryIO, algo_inst=None):
    """
    Compute the hash of a given file using algo, or update the contents of algo_inst if provided
    :param fileobj: file object that would be used
    :param algo_inst: instance of hashlib object. If None then md5 from hashlib is used.
    :returns new algo instance, or reference to algo_inst if provided
    """
    if algo_inst is None:
        algo_inst = md5()
    map = mmap.mmap(fileobj.fileno(), 0, prot=mmap.PROT_READ)
    algo_inst.update(map)
    map.close()
    return algo_inst


def find_free_filename(dir: str, fname: str, pattern_sym="#", try_num: Iterable = None) -> Tuple[str, object]:
    assert pattern_sym in fname
    if not isinstance(try_num, Iterable):
        try_num = itertools.count()

    files = os.listdir(dir)

    for i in try_num:
        ftry = fname.replace(pattern_sym, f"{i}")
        if ftry in files:
            continue
        else:
            return ftry, i
    else:
        raise FileExistsError("Could ont find suitable name!!")


def set_process_name(name):
    """
    Change process name for killall
    @param name: name to set
    @return: True if successful, false if not
    """
    try:
        global _libc
        if _libc is None:
            import ctypes
            _libc = ctypes.CDLL('libc.so.6')
        _libc.prctl(15, name, 0, 0, 0)
        return True
    except Exception as e:
        warnings.warn('Unable to set processName: %s" % e')
        return False


def _get_xdg_path(resourceType):
    if BaseDirectory is not None:
        if resourceType == "data":
            base = BaseDirectory.xdg_data_home
            if base == "/usr/share/mime":
                # Ugly hack because somehow Maemo 4.1 seems to be set to this
                base = None
        elif resourceType == "config":
            base = BaseDirectory.xdg_config_home
        elif resourceType == "cache":
            base = BaseDirectory.xdg_cache_home
        else:
            raise RuntimeError("Unknown type: " + resourceType)
    else:
        base = None

    return base


def get_resource_path(resourceType, resource, name=None):
    base = _get_xdg_path(resourceType)
    if base is not None:
        dirPath = os.path.join(base, resource)
    else:
        base = os.path.join(os.path.expanduser("~"), ".%s" % resource)
        dirPath = base
    if name is not None:
        dirPath = os.path.join(dirPath, name)
    return dirPath


def get_new_resource(resourceType, resource, name):
    dirPath = get_resource_path(resourceType, resource)
    filePath = os.path.join(dirPath, name)
    if not os.path.exists(dirPath):
        # Looking before I leap to not mask errors
        os.makedirs(dirPath)

    return filePath


def get_existing_resource(resourceType, resource, name):
    base = _get_xdg_path(resourceType)

    if base is not None:
        finalPath = os.path.join(base, name)
        if os.path.exists(finalPath):
            return finalPath

    altBase = os.path.join(os.path.expanduser("~"), ".%s" % resource)
    finalPath = os.path.join(altBase, name)
    if os.path.exists(finalPath):
        return finalPath
    else:
        raise RuntimeError("Resource not found: %r" % ((resourceType, resource, name),))


def safe_remove(target):
    try:
        os.remove(target)
    except OSError as e:
        if e.errno == 2:
            pass
        else:
            raise e


def check_ram(threshold_percent=5, exit_on_failure=True):
    availram = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    if availram < threshold_percent:
        print(f"RAMCHECK {availram}% left, threshold {threshold_percent}%.")
        if exit_on_failure:
            exit(1)
        else:
            raise MemoryError("System memory low")


def signal_pdb(sig, frame):
    import pdb
    print(f"Caught signal {sig} at frame {frame} for PDB, starting trace")
    pdb.Pdb().set_trace(frame)


class Test_lib_linux(unittest.TestCase):
    def test_check_ram(self):
        with self.assertRaises(MemoryError):
            check_ram(threshold_percent=100, exit_on_failure=False)

    def test_get_file_md5(self):
        file = os.path.abspath(__file__)
        fileobj = open(file, 'rb')
        hash1 = get_file_hash(fileobj).hexdigest()
        self.assertEqual(fileobj.tell(), 0) # make sure file pointer was not messed with

        # compute hash with md5sum and verify
        import subprocess
        hash2 = subprocess.getoutput(f"md5sum {file}")
        self.assertTrue(hash2.startswith(hash1)) #


if __name__ == '__main__':
    unittest.main()