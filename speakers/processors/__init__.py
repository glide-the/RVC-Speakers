"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from speakers.common.registry import registry
from speakers.processors.base_processor import BaseProcessor
from speakers.processors.base_processor import ProcessorData
from speakers.processors.rvc_speakers_processor import RvcProcessorData

__all__ = [
    "BaseProcessor",
    "ProcessorData",
    "RvcProcessorData",
]


def load_preprocess(config: dict = None):
    """
    Load preprocessor configs and construct preprocessors.

    If no preprocessor is specified, return BaseProcessor, which does not do any preprocessing.

    Args:
        config (dict): preprocessor configs.

    Returns:
        vits_processors (dict): preprocessors for vits inputs.
        rvc_processors (dict): preprocessors for rvc inputs.

    """

    if config is None:
        raise RuntimeError("Load preprocessor configs is None.")

    def _build_proc_from_cfg(cfg):
        print(cfg)
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else BaseProcessor()
        )

    # vits_proc_cfg = config.get("vits_processor")
    rvc_proc_cfg = config.get("rvc_processor")

    # vits_processors = _build_proc_from_cfg(vits_proc_cfg)
    vits_processors = None

    rvc_processors = _build_proc_from_cfg(rvc_proc_cfg)

    return vits_processors, rvc_processors
