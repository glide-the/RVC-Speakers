import os
from typing import List, Callable, Tuple
import numpy as np
import hashlib
import re

MODULE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
BASE_PATH = os.path.dirname(MODULE_PATH)


# Adapted from argparse.Namespace
class Context(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def __eq__(self, other):
        if not isinstance(other, Context):
            return NotImplemented
        return vars(self) == vars(other)

    def __contains__(self, key):
        return key in self.keys()

    def __repr__(self):
        type_name = type(self).__name__
        arg_strings = []
        star_args = {}
        for arg in self._get_args():
            arg_strings.append(repr(arg))
        for name, value in self._get_kwargs():
            if name.isidentifier():
                arg_strings.append('%s=%r' % (name, value))
            else:
                star_args[name] = value
        if star_args:
            arg_strings.append('**%s' % repr(star_args))
        return '%s(%s)' % (type_name, ', '.join(arg_strings))

    def _get_kwargs(self):
        return list(self.items())

    def _get_args(self):
        return []


def repeating_sequence(s: str):
    """Extracts repeating sequence from string. Example: 'abcabca' -> 'abc'."""
    for i in range(1, len(s) // 2 + 1):
        seq = s[:i]
        if seq * (len(s) // len(seq)) + seq[:len(s) % len(seq)] == s:
            return seq
    return s


def replace_prefix(s: str, old: str, new: str):
    if s.startswith(old):
        s = new + s[len(old):]
    return s


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_digest(file_path: str) -> str:
    h = hashlib.sha256()
    BUF_SIZE = 65536

    with open(file_path, 'rb') as file:
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = file.read(BUF_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_filename_from_url(url: str, default: str = '') -> str:
    m = re.search(r'/([^/?]+)[^/]*$', url)
    if m:
        return m.group(1)
    return default


def prompt_yes_no(query: str, default: bool = None) -> bool:
    s = '%s (%s/%s): ' % (query, 'Y' if default == True else 'y', 'N' if default == False else 'n')
    while True:
        inp = input(s).lower()
        if inp in ('yes', 'y'):
            return True
        elif inp in ('no', 'n'):
            return False
        elif default != None:
            return default
        if inp:
            print('Error: Please answer with "y" or "n"')

