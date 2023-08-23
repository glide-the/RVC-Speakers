
from speakers.common.registry import registry
import os

root_dir = os.path.dirname(os.path.abspath(__file__))
registry.register_path("bark_library_root", root_dir)

