from abc import abstractmethod
from typing import List, Dict

from speakers.load.serializable import Serializable
from speakers.processors import ProcessorData, BaseProcessor
from collections import deque

import logging

class FlowData(Serializable):
    """
    当前runner的任务参数
    """

    @property
    @abstractmethod
    def type(self) -> str:
        """Type of the Message, used for serialization."""
    @property
    def lc_serializable(self) -> bool:
        """Whether this class is Processor serializable."""
        return True


class Runner(Serializable):
    """ runner的任务id"""
    task_id: str
    flow_data: FlowData

    @property
    def type(self) -> str:
        """Type of the Runner Message, used for serialization."""
        return 'runner'

    @property
    def lc_serializable(self) -> bool:
        """Whether this class is Processor serializable."""
        return True


# Define a base class for tasks
class BaseTask:
    """
        基础任务处理器由任务管理器创建，用于执行runner flow 的任务，子类实现具体的处理流程
        此类定义了流程runner task的生命周期
    """

    def __init__(self, preprocess_dict: Dict[str, BaseProcessor]):
        self._progress_hooks = []
        self._add_logger_hook()
        self._preprocess_dict = preprocess_dict
        self.logger = logging.getLogger('base_task_runner')

    @classmethod
    def from_config(cls, cfg=None):
        return cls(preprocess_dict={})

    def _add_logger_hook(self):
        LOG_MESSAGES = {
            'voice_task': 'Running voice_task',
            'saved': 'Saving results',
        }
        LOG_MESSAGES_SKIP = {
            'skip-no-text': 'No text regions with text! - Skipping',
        }
        LOG_MESSAGES_ERROR = {
            'error': 'task error',
        }

        async def ph(state, finished):
            if state in LOG_MESSAGES:
                self.logger.info(LOG_MESSAGES[state])
            elif state in LOG_MESSAGES_SKIP:
                self.logger.warn(LOG_MESSAGES_SKIP[state])
            elif state in LOG_MESSAGES_ERROR:
                self.logger.error(LOG_MESSAGES_ERROR[state])

        self.add_progress_hook(ph)

    def add_progress_hook(self, ph):
        self._progress_hooks.append(ph)

    async def report_progress(self, state: str, finished: bool = False):
        for ph in self._progress_hooks:
            await ph(state, finished)

    @classmethod
    def prepare(cls, runner: Runner):
        """
        预处理

        Args:
            runner (Runner): runner flow data

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError

    @classmethod
    async def dispatch(cls, runner: Runner):
        """
        当前runner task具体flow data

        Args:
            runner (ProcessorData): runner flow data

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError

    @classmethod
    def complete(cls, runner: Runner):
        """
        后置处理

        Args:
            runner (Runner): runner flow data

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError
