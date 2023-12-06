from collections import deque
from typing import Dict, List
from speakers.server.model.flow_data import BaseFlowData


class Bootstrap:

    """Used by web module to decide which secret for securing"""
    _NONCE: str = ''
    """最大的任务队列"""
    _MAX_ONGOING_TASKS: int = 1

    """任务队列"""
    _QUEUE: deque = deque()
    """进行的任务数据"""
    _TASK_DATA: Dict[str, BaseFlowData] = {}
    """进行的任务状态"""
    _TASK_STATES = {}
    """正在进行的任务"""
    _ONGOING_TASKS: List[str] = []

    def __init__(self):
        self._version = "v0.0.1"

    @classmethod
    def from_config(cls, cfg=None):
        return cls()

    @property
    def version(self):
        return self._version

    @property
    def max_ongoing_tasks(self) -> int:
        return self._MAX_ONGOING_TASKS

    @property
    def ongoing_tasks(self) -> List[str]:
        return self._ONGOING_TASKS

    @property
    def queue(self) -> deque:
        return self._QUEUE

    @property
    def task_data(self) -> Dict[str, BaseFlowData]:
        return self._TASK_DATA

    @property
    def task_states(self) -> dict:
        return self._TASK_STATES

    @property
    def nonce(self) -> str:
        return self._NONCE

    def set_nonce(self, nonce: str):
        self._NONCE = nonce

    @classmethod
    async def run(cls):
        raise NotImplementedError
