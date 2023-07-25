import torch

import util

device = (
    'cuda:0' if torch.cuda.is_available()
    else (
        'mps' if util.has_mps()
        else 'cpu'
    )
)
is_half = util.is_half(device)

x_pad = 3 if is_half else 1
x_query = 10 if is_half else 6
x_center = 60 if is_half else 38
x_max = 65 if is_half else 41
