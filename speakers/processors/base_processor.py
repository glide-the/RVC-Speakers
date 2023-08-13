"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from omegaconf import OmegaConf
from abc import abstractmethod
from speakers.load.serializable import Serializable


class ProcessorData(Serializable):
    """
    The base abstract ProcessorData class.
    """

    @property
    @abstractmethod
    def type(self) -> str:
        """Type of the Message, used for serialization."""

    @property
    def lc_serializable(self) -> bool:
        """Whether this class is Processor serializable."""
        return True


class BaseProcessor:
    """
    音频处理器有抽象处理器Processor，通过单独的Processor配置，
    通过from_config工厂方法预加载音频处理器
    """
    def __init__(self):
        self.transform = lambda x: x
        return

    def __call__(self, data: ProcessorData):
        return self.transform(data)

    @classmethod
    def from_config(cls, cfg=None):
        return cls()

    def build(self, **kwargs):
        cfg = OmegaConf.create(kwargs)

        return self.from_config(cfg)

