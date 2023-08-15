from typing import List

from speakers.processors import ProcessorData, BaseProcessor
from collections import deque


class FlowData:
    """
    当前runner的任务参数
    """


class Runner:
    """ runner的任务id"""
    task_id: str
    flow_data: FlowData
    processor_q: deque[str]


# Define a base class for tasks
class BaseTask:
    """
        基础任务处理器由任务管理器创建，用于执行runner flow 的任务，子类实现具体的处理流程
        此类定义了流程runner task的生命周期
    """

    def __init__(self, preprocess_dict: [str, BaseProcessor]):
        self.preprocess_dict = preprocess_dict

    @classmethod
    def from_config(cls, cfg=None):
        return cls(preprocess_dict={})

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
    def dispatch(cls, runner: Runner):
        """
        当前runner task具体flow data

        Args:
            runner (ProcessorData): runner flow data

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError

    @classmethod
    def support(cls, runner: Runner):
        """
        用于检测当前runner task中用到的 processor是否支持

        Args:
            runner (Runner): 校测 runner flow data 传入processor名称列表

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
