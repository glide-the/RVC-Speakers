from typing import Dict
from speakers.processors import ProcessorData, BaseProcessor, get_processors, VitsProcessorData, RvcProcessorData
from speakers.tasks import AudioTaskAbstract, Runner, FlowData
from speakers.common.registry import registry
from speakers.server.model.flow_data import PayLoad
import traceback
import hashlib


class VitsVoiceFlowData(FlowData):
    vits: VitsProcessorData
    rvc: RvcProcessorData

    @property
    def type(self) -> str:
        """Type of the FlowData Message, used for serialization."""
        return "vits_voice"


def calculate_md5(input_string):
    md5_hash = hashlib.md5()
    md5_hash.update(input_string.encode('utf-8'))
    return md5_hash.hexdigest()


@registry.register_task("vits_voice_task")
class VitsVoiceTask(AudioTaskAbstract):
    SAMPLE_RATE: int = 22050

    def __init__(self, preprocess_dict: Dict[str, BaseProcessor]):
        super().__init__(preprocess_dict=preprocess_dict)
        self._preprocess_dict = preprocess_dict

    @classmethod
    def from_config(cls, cfg=None):
        preprocess_dict = {}
        for preprocess in cfg.get('preprocess'):
            for key, preprocess_info in preprocess.items():
                preprocess_object = get_processors(preprocess_info.processor)
                preprocess_dict[preprocess_info.processor_name] = preprocess_object

        return cls(preprocess_dict=preprocess_dict)

    @property
    def preprocess_dict(self) -> Dict[str, BaseProcessor]:
        return self._preprocess_dict

    @classmethod
    def prepare(cls, payload: PayLoad) -> Runner:
        """
        runner任务构建
        """
        params = payload.payload
        # 获取payload中的vits和rvc的值
        vits_data = params.get("vits", {})
        rvc_data = params.get("rvc", {})

        # noise_scale_w: noise_scale_w(控制音素发音长度)
        noise_scale_w = vits_data.get("noise_scale_w")
        # noise_scale(控制感情变化程度)
        noise_scale = vits_data.get("noise_scale")
        # length_scale(控制整体语速)
        speed = vits_data.get("speed")
        # 语言
        language = vits_data.get("language")
        # vits 讲话人
        speaker_id = vits_data.get("speaker_id")
        text = vits_data.get("text")

        # 创建一个 VitsProcessorData 实例
        vits_processor_data = VitsProcessorData(
            text=text,
            language=language,
            speaker_id=speaker_id,
            noise_scale=noise_scale,
            speed=speed,
            noise_scale_w=noise_scale_w
        )
        # 获取rvc中的值

        model_index = rvc_data.get("model_index")

        # 变调(整数, 半音数量, 升八度12降八度-12)
        f0_up_key = rvc_data.get("f0_up_key")
        f0_method = rvc_data.get("f0_method")

        # 检索特征占比
        index_rate = rvc_data.get("index_rate")
        # >=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音
        filter_radius = rvc_data.get("filter_radius")
        # 输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络
        rms_mix_rate = rvc_data.get("rms_mix_rate")
        # 后处理重采样至最终采样率，0为不进行重采样
        resample_rate = rvc_data.get("resample_sr")

        rvc_protect = rvc_data.get("protect")
        rvc_f0_file = rvc_data.get("f0_file")

        rvc_processor_data = RvcProcessorData(
            model_index=model_index,
            f0_up_key=f0_up_key,
            f0_method=f0_method,
            index_rate=index_rate,
            filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate,
            resample_sr=resample_rate,
            f0_file=rvc_f0_file,
            protect=rvc_protect
        )

        # 创建一个 VitsVoiceFlowData 实例，并将 VitsProcessorData 实例作为参数传递
        voice_flow_data = VitsVoiceFlowData(vits=vits_processor_data,
                                        rvc=rvc_processor_data)

        # 创建 Runner 实例并传递上面创建的 VitsVoiceFlowData 实例作为参数
        task_id = f'{calculate_md5(text)}-{speaker_id}-{language}' \
                  f'-{noise_scale}-{speed}-{noise_scale_w}' \
                  f'-{model_index}-{f0_up_key}'
        runner = Runner(
            task_id=task_id,
            flow_data=voice_flow_data
        )

        return runner

    async def dispatch(self, runner: Runner):

        try:
            # 加载task
            self.logger.info('dispatch')

            # 开启任务1
            await self.report_progress(task_id=runner.task_id, runner_stat='vits_voice_task', state='dispatch_vits_voice_task')
            data = runner.flow_data
            if 'vits_voice' in data.type:
                if 'VITS' in data.vits.type:
                    vits_preprocess_object = self.preprocess_dict.get(data.vits.type)
                    if not vits_preprocess_object.match(data.vits):
                        raise RuntimeError('不支持的process')
                    audio_np = vits_preprocess_object(data.vits)
                    if audio_np is not None and 'RVC' in data.rvc.type:
                        # 将 NumPy 数组转换为 Python 列表
                        audio_samples_list = audio_np.tolist()
                        data.rvc.sample_rate = self.SAMPLE_RATE
                        data.rvc.audio_samples = audio_samples_list
                        rvc_preprocess_object = self.preprocess_dict.get(data.rvc.type)
                        if not rvc_preprocess_object.match(data.rvc):
                            raise RuntimeError('不支持的process')

                        write_data = rvc_preprocess_object(data.rvc)

                        # 完成任务，构建响应数据
                        await self.report_progress(task_id=runner.task_id,
                                                   runner_stat='vits_voice_task',
                                                   state='finished',
                                                   finished=True)

                        del audio_np
                        await super().save_task_write(runner=runner, write_data=write_data)
                        del runner

        except Exception as e:
            await self.report_progress(task_id=runner.task_id, runner_stat='vits_voice_task',
                                       state='error', finished=True)

            self.logger.error(f'{e.__class__.__name__}: {e}',
                              exc_info=e)

            traceback.print_exc()

        return None, None

    def complete(self, runner: Runner):
        pass
