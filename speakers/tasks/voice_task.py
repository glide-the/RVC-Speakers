from typing import Dict

from speakers.processors import ProcessorData, BaseProcessor, get_processors, VitsProcessorData, RvcProcessorData
from speakers.tasks import BaseTask, Runner, FlowData
from speakers.common.registry import registry


class VoiceFlowData(FlowData):
    vits: VitsProcessorData
    rvc: RvcProcessorData

    @property
    def type(self) -> str:
        """Type of the FlowData Message, used for serialization."""
        return "voice"


@registry.register_task("voice_task")
class VoiceTask(BaseTask):

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

    def prepare(self, runner: Runner):
        pass

    def dispatch(self, runner: Runner):
        data = runner.flow_data
        if 'voice' in data.type:
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

                    out_sr, output_audio = rvc_preprocess_object(data.rvc)

                    del audio_np
                    del runner
                    return out_sr, output_audio

        return None

    def support(self, runner: Runner) -> bool:
        for type_name in runner.processor_q:
            if self.preprocess_dict.get(type_name) is None:
                raise NotImplementedError

        return True

    def complete(self, runner: Runner):
        pass
