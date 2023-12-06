import asyncio
import logging
import os
import traceback

import requests
from omegaconf import OmegaConf

from speakers.common.log import add_file_logger, remove_file_logger
from speakers.common.registry import registry
from speakers.common.utils import get_abs_path, get_tmp_path
from speakers.processors import (load_preprocess
                                 )
from speakers.server.model.flow_data import PayLoad
from speakers.tasks import get_task, load_task, tasks_cache

logger = logging.getLogger('speaker_runner')


def set_main_logger(l):
    global logger
    logger = l


class Speaker:
    """
        在任务系统中，"runner" 通常指的是负责执行任务的组件或模块。Runner 的生命周期是指它从创建到销毁的整个过程，
        其中包含了一些关键的阶段和操作。虽然不同的任务系统可能会有不同的实现细节，但通常可以概括为以下几个主要的生命周期阶段：

        创建（Creation）：在任务被提交或调度之前，runner 需要被创建。这个阶段可能涉及资源分配、
        初始化操作、配置设置等。Runner 的创建过程通常包括为任务分配所需的资源，准备运行环境，建立与任务管理器或调度器的连接等。

        准备（Preparation）：在 runner 开始执行任务之前，需要对任务本身进行一些准备工作。这可能包括下载所需的文件、
        加载依赖项、设置环境变量、配置运行参数等。准备阶段的目标是确保任务在执行过程中所需的一切都已就绪。

        执行（Execution）：执行阶段是 runner 的主要功能，它负责实际运行任务的代码逻辑。
        这可能涉及到执行计算、处理数据、调用外部服务等，具体取决于任务的性质。在执行阶段，runner 需要监控任务的进展，
        处理异常情况，并可能与其他系统组件进行交互。

        监控与报告（Monitoring and Reporting）：在任务执行期间，runner 需要持续监控任务的状态和进度。
        这可能包括收集性能指标、记录日志、报告错误或异常等。监控与报告是确保任务在预期范围内执行的重要手段。

        完成与清理（Completion and Cleanup）：当任务执行结束后，runner 需要处理任务的完成操作。
        这可能涉及处理执行结果、释放资源、清理临时文件、关闭连接等。完成与清理阶段的目标是确保任务执行后不会留下不必要的残留状态。

        销毁（Destruction）：在 runner 的生命周期结束时，
        需要进行销毁操作。这可能包括释放占用的资源、关闭连接、清理临时数据等。销毁阶段的目标是确保系统不会因为不再需要的 runner 而产生资源浪费或不稳定性。

    """

    def __init__(self, speakers_config_file: str = 'speakers.yaml',
                 verbose: bool = False):
        self.verbose = verbose

        config = OmegaConf.load(get_abs_path(speakers_config_file))
        load_preprocess(config=config.get('preprocess'))
        load_task(config.get("tasks"))

    async def preparation_runner(self, task_id: str, payload: PayLoad = None):
        voice_task = get_task(payload.parameter.task_name)
        try:

            runner = voice_task.prepare(payload=payload)

            await voice_task.dispatch(runner=runner)

            await voice_task.report_progress(task_id=runner.task_id, runner_stat='preparation_runner',
                                             state='end',
                                             finished=True)
        except Exception as e:

            logger.error(f'{e.__class__.__name__}: {e}',
                         exc_info=e)
            await voice_task.report_progress(task_id=task_id, runner_stat='preparation_runner',
                                             state='error', finished=True)


class WebSpeaker(Speaker):
    def __init__(self, speakers_config_file: str = 'speakers.yaml',
                 verbose: bool = False,
                 nonce: str = ''):
        super().__init__(speakers_config_file=speakers_config_file, verbose=verbose)

        config = OmegaConf.load(get_abs_path(speakers_config_file))
        remote_infos = {}
        for bootstraps in config.get("bootstrap"):
            for key, bootstrap_cfg in bootstraps.items():  # 使用 .items() 方法获取键值对

                remote_infos[bootstrap_cfg.name] = {
                    'host': bootstrap_cfg.host,
                    'port': bootstrap_cfg.port
                }

        self.remote_infos = remote_infos
        self.nonce = nonce
        self._task_results = {}

    async def listen(self):
        """
        监听server端任务，注册任务监听器接收消息通知
        """
        logger.info('Waiting for WebSpeaker tasks')

        async def sync_state(task_id: str, runner_stat: str, state: str, finished: bool, result: dict):
            # wait for translation to be saved first (bad solution?)
            finished = finished and not state == 'finished'
            while True:
                try:
                    data = {
                        'task_id': task_id,
                        'runner_stat': runner_stat,
                        'nonce': self.nonce,
                        'state': state,
                        'finished': finished,
                        'result': result
                    }
                    # 处理每个runner的调度
                    for _key, _remote_info in self.remote_infos.items():

                        if self._task_results.get(_key) is None or self._task_results.get(_key).get("task_id") is None:
                            continue
                        if task_id in self._task_results.get(_key).get("task_id"):
                            host = '127.0.0.1'
                            if not '0.0.0.0' in _remote_info.get("host"):
                                host = _remote_info.get("host")

                            requests.post(
                                f'http://{host}:{_remote_info.get("port")}/runner/task-update-internal',
                                json=data, timeout=20)
                            break
                    break
                except Exception:
                    # if translation is finished server has to know
                    if finished:
                        continue
                    else:
                        break

        for key, task in tasks_cache.items():
            task.add_progress_hook(sync_state)

        while True:
            self._task_results = self._get_task()

            wait_flag = False

            if not self._task_results:
                wait_flag = True

            if wait_flag:
                await asyncio.sleep(1)
                continue

            # 处理每个runner的调度,如果有任何一个返回了任务，则执行调度
            for key, remote_info in self.remote_infos.items():
                # TODO 此处需要分布式调度，需要考虑重复调度的问题
                if self._task_results.get(key) is None or self._task_results.get(key).get("task_id") is None:
                    wait_flag = True
                else:
                    wait_flag = False
                    break

            if wait_flag:
                await asyncio.sleep(1)
                continue

            # if self.verbose:
            #     # Write log file
            #     log_file = self._result_path('log.txt')
            #     add_file_logger(log_file)

            # TODO 调度任务应当从队列中获取，而不是从runner_bootstrap_web中获取
            # 处理每个runner的调度
            for key, remote_info in self.remote_infos.items():
                logger.info(f'Processing task {self._task_results.get(key).get("task_id")}')

                await self.preparation_runner(task_id=self._task_results.get(key).get("task_id"),
                                              payload=PayLoad.parse_obj(self._task_results.get(key).get("data")))

            # if self.verbose:
            #     # Write log file
            #     log_file = self._result_path('log.txt')
            #     remove_file_logger(log_file)

    def _get_task(self):
        try:
            task_results = {}
            for key, _remote_info in self.remote_infos.items():  # 使用 .items() 方法获取键值对

                host = '127.0.0.1'
                if not '0.0.0.0' in _remote_info.get("host"):
                    host = _remote_info.get("host")

                response = requests.get(
                    f'http://{host}:{_remote_info["port"]}/runner/task-internal?nonce={self.nonce}',
                    timeout=3600)
                # 检查响应状态码
                if response.status_code == 200:
                    task_results[key] = response.json().get("data")
            return task_results
        except Exception:
            logger.error(f'runner_bootstrap_web connection error: {traceback.format_exc()}')
            return None
