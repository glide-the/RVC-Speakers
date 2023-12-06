from speakers.common.registry import registry
import os
from TransNetV2.modules import TransNetV2
from TransNetV2.model_load import TransNetV2ModelLoader

root_dir = os.path.dirname(os.path.abspath(__file__))
registry.register_path("TransNetV2_library_root", root_dir)

__all__ = [
    "TransNetV2",
    "TransNetV2ModelLoader"
]
