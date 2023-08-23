"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from typing import List

from speakers.common.registry import registry
from speakers.processors.base_processor import BaseProcessor
from speakers.processors.base_processor import ProcessorData
from speakers.processors.rvc_speakers_processor import RvcProcessorData
from speakers.processors.vits_to_voice import VitsProcessorData
from speakers.processors.bark_to_voice import BarkProcessorData

__all__ = [
    "BaseProcessor",
    "ProcessorData",
    "RvcProcessorData",
    "VitsProcessorData",
    "BarkProcessorData",
    "get_processors",
    "load_preprocess",
]

processors_cache = {}


def load_preprocess(config: List[dict] = None):
    """
    Load preprocessor configs and construct preprocessors.

    If no preprocessor is specified, return BaseProcessor, which does not do any preprocessing.

    Args:
        config (List[dict]): preprocessor configs.

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
    for process_cfg in config:  # 使用 .items() 方法获取键值对
        for key, processor_cfg in process_cfg.items():  # 使用 .items() 方法获取键值对
            processors = _build_proc_from_cfg(processor_cfg)
            processors_cache[key] = processors


def get_processors(key: str) -> BaseProcessor:

    if not processors_cache.get(key):
        raise ValueError(f'Could not find processors for: "{key}". '
                         f'Choose from the following: %s' % ','.join(processors_cache))

    return processors_cache[key]
