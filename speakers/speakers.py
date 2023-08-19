import numpy as np
import logging
import os
import requests
import asyncio
from omegaconf import OmegaConf

from speakers.common.registry import registry
from speakers.processors import (load_preprocess,
                                 RvcProcessorData,
                                 get_processors,
                                 VitsProcessorData
                                 )
from speakers.common.utils import get_abs_path
from speakers.tasks import get_task, load_task
from speakers.tasks import VoiceFlowData, Runner

from scipy.io.wavfile import write as write_wav

from speakers.common.log import add_file_logger, remove_file_logger
from speakers.server.model.flow_data import VoiceFlowData as VoiceFlow

import traceback

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

    async def preparation_runner(self, task_id: str, params: VoiceFlow = None):
        """
        任务构建
        """

        vits_voice = get_processors('vits_processor')
        # noise_scale_w: noise_scale_w(控制音素发音长度)
        noise_scale_w = params.vits.noise_scale_w
        # noise_scale(控制感情变化程度)
        noise_scale = params.vits.noise_scale
        # length_scale(控制整体语速)
        speed = params.vits.speed
        # 语言
        language = params.vits.language
        # vits 讲话人
        speaker_id = params.vits.speaker_id
        text = params.vits.text

        # 创建一个 VitsProcessorData 实例
        vits_processor_data = VitsProcessorData(
            text=text,
            language=language,
            speaker_id=speaker_id,
            noise_scale=noise_scale,
            speed=speed,
            noise_scale_w=noise_scale_w
        )

        model_index = params.rvc.model_index

        # 变调(整数, 半音数量, 升八度12降八度-12)
        f0_up_key = params.rvc.f0_up_key
        f0_method = params.rvc.f0_method

        # 检索特征占比
        index_rate = params.rvc.index_rate
        # >=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音
        filter_radius = params.rvc.filter_radius
        # 输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络
        rms_mix_rate = params.rvc.rms_mix_rate
        # 后处理重采样至最终采样率，0为不进行重采样
        resample_rate = params.rvc.resample_sr

        rvc_processor_data = RvcProcessorData(
            model_index=model_index,
            f0_up_key=f0_up_key,
            f0_method=f0_method,
            index_rate=index_rate,
            filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate,
            resample_sr=resample_rate
        )

        # 创建一个 VoiceFlowData 实例，并将 VitsProcessorData 实例作为参数传递
        voice_flow_data = VoiceFlowData(vits=vits_processor_data,
                                        rvc=rvc_processor_data)

        # 创建 Runner 实例并传递上面创建的 VoiceFlowData 实例作为参数

        runner = Runner(
            task_id=task_id,
            flow_data=voice_flow_data
        )
        voice_task = get_task("voice_task")

        out_sr, output = await voice_task.dispatch(runner=runner)
        if output is not None:
            # 当前生命周期 saved
            # save audio to disk
            write_wav(self._result_path(f"{task_id}.wav"), out_sr, output)
            del output
            await voice_task.report_progress(task_id=runner.task_id, runner_stat='voice_task', state='saved',
                                             finished=True)

    def _result_path(self, path: str) -> str:
        return os.path.join(registry.get_path("library_root"), 'result', path)


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

        async def sync_state(state: str, finished: bool):
            # wait for translation to be saved first (bad solution?)
            finished = finished and not state == 'finished'
            while True:
                try:
                    data = {
                        'task_id': self._task_id,
                        'nonce': self.nonce,
                        'state': state,
                        'finished': finished,
                    }
                    requests.post(f'http://{self.host}:{self.port}/task-update-internal', json=data, timeout=20)
                    break
                except Exception:
                    # if translation is finished server has to know
                    if finished:
                        continue
                    else:
                        break

        voice_task = get_task("voice_task")
        voice_task.add_progress_hook(sync_state)

        while True:
            self._task_results = self._get_task()
            if not self._task_results or self._task_results.get("task_id") is None:
                await asyncio.sleep(1)
                continue

            logger.info(f'Processing task {self._task_results.get("task_id")}')

            if self.verbose:
                # Write log file
                log_file = self._result_path('log.txt')
                add_file_logger(log_file)

            await self.preparation_runner(task_id=self._task_results.get("task_id"),
                                          params=self._task_results.get("data"))

            if self.verbose:
                # Write log file
                log_file = self._result_path('log.txt')
                remove_file_logger(log_file)

    def _get_task(self):
        try:
            task_results = {}
            for key, remote_info in self.remote_infos.items():  # 使用 .items() 方法获取键值对

                response = requests.get(
                    f'http://{remote_info["host"]}:{remote_info["port"]}/runner/task-internal?nonce={self.nonce}',
                    timeout=3600)
                # 检查响应状态码
                if response.status_code == 200:
                    task_results[key] = response.json().get("data")
                else:
                    raise RuntimeError
            return task_results
        except Exception:
            traceback.print_exc()
            return None
