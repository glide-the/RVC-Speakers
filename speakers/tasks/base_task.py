from abc import abstractmethod
from typing import List, Dict

from speakers.load.serializable import Serializable
from speakers.processors import ProcessorData, BaseProcessor
from speakers.server.model.flow_data import PayLoad

from typing import Tuple
from speakers.common.utils import get_abs_path, get_tmp_path
from scipy.io.wavfile import write as write_wav
import logging
import numpy as np


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
        """
        默认的任务日志监听者
        :return:
        """
        LOG_MESSAGES = {
            'dispatch_voice_task': 'dispatch_voice_task',
            'saved': 'Saving results',
        }
        LOG_MESSAGES_SKIP = {
            'skip-no-text': 'No text regions with text! - Skipping',
        }
        LOG_MESSAGES_ERROR = {
            'error': 'task error',
        }

        async def ph(task_id: str, runner_stat: str, state: str, finished: bool = False, result: dict = {}):
            if state in LOG_MESSAGES:
                self.logger.info(LOG_MESSAGES[state])
            elif state in LOG_MESSAGES_SKIP:
                self.logger.warn(LOG_MESSAGES_SKIP[state])
            elif state in LOG_MESSAGES_ERROR:
                self.logger.error(LOG_MESSAGES_ERROR[state])

        self.add_progress_hook(ph)

    def add_progress_hook(self, ph):
        """
        注册监听器
        :param ph: 监听者
        :return:
        """
        self._progress_hooks.append(ph)

    async def report_progress(self, task_id: str, runner_stat: str, state: str, finished: bool = False,
                              result: dict = {}):
        """
        任务通知监听器
        :param result:
        :param task_id: 任务id
        :param runner_stat: 任务执行位置
        :param state: 状态
        :param finished: 是否完成
        :return:
        """
        for ph in self._progress_hooks:
            await ph(task_id=task_id, runner_stat=runner_stat, state=state, finished=finished, result=result)

    @classmethod
    def prepare(cls, payload: PayLoad) -> Runner:
        """
        预处理

        Args:
            payload (PayLoad): runner flow data

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError

    @classmethod
    async def dispatch(cls, runner: Runner) -> None:
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


class AudioTaskAbstract(BaseTask):
    """
    抽象的音频处理，用来处理音频保存逻辑和任务生命周期结束
    """

    async def save_task_write(self, runner: Runner, write_data: Tuple[int, np.ndarray]):
        """
        抽象处理分配
        用来处理音频保存逻辑和任务生命周期结束
        :param write_data:
        :param runner:
        :return:
        """
        out_sr, output_audio = write_data
        if output_audio is not None:
            # 当前生命周期 saved
            # save audio to disk
            write_wav(self._result_path(f"{runner.task_id}.wav"), out_sr, output_audio)
            del output_audio
            await self.report_progress(task_id=runner.task_id, runner_stat='save_task_write',
                                       state='save_write',
                                       finished=False,
                                       result={
                                           'filename': self._result_path(f"{runner.task_id}.wav")
                                       })

    def _result_path(self, path: str) -> str:
        return get_tmp_path(f'result/{path}')
