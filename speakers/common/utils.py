from speakers.common.registry import registry
import os


def get_abs_path(rel_path):
    return os.path.join(registry.get_path("library_root"), rel_path)
