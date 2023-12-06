from speakers.common.registry import registry
import os


def get_abs_path(rel_path):
    return os.path.join(registry.get_path("library_root"), rel_path)


def get_tmp_path(rel_path):
    return os.path.join(registry.get_path("tmp_root"), rel_path)
