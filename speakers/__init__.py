from speakers.common.registry import registry
from speakers.speakers import set_main_logger, Speaker, WebSpeaker
import torch
import os
import util
from pathlib import Path
import platform
import tempfile

__all__=[
    "Speaker",
    "WebSpeaker",
    "set_main_logger",
]

root_dir = os.path.dirname(os.path.abspath(__file__))
registry.register_path("library_root", root_dir)

tempdir = Path("/tmp" if platform.system() == "Darwin" else tempfile.gettempdir())
registry.register_path("tmp_root", str(tempdir))


device = (
    'cuda:0' if torch.cuda.is_available()
    else (
        'mps' if util.has_mps()
        else 'cpu'
    )
)

registry.register("device", device)

is_half = util.is_half(device)

registry.register("is_half", is_half)

x_pad = 3 if is_half else 1
x_query = 10 if is_half else 6
x_center = 60 if is_half else 38
x_max = 65 if is_half else 41

registry.register("x_pad", x_pad)
registry.register("x_query", x_query)
registry.register("x_center", x_center)
registry.register("x_max", x_max)
