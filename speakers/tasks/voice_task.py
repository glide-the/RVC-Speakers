
from speakers.processors import ProcessorData, BaseProcessor, get_processors
from speakers.tasks import BaseTask, Runner
from speakers.common.registry import registry


@registry.register_task("voice_task")
class VoiceTask(BaseTask):

    def __init__(self, preprocess_dict: [str, BaseProcessor]):
        super().__init__(preprocess_dict)
        self.preprocess_dict = preprocess_dict

    @classmethod
    def from_config(cls, cfg=None):
        preprocess_dict = {}
        for preprocess in cfg.get('preprocess'):
            for key, preprocess_info in preprocess.items():
                preprocess_object = get_processors(preprocess_info.processor)
                preprocess_dict[preprocess_info.processor_name] = preprocess_object

        return cls(preprocess_dict=preprocess_dict)

    @property
    def preprocess_dict(self) -> [str, BaseProcessor]:
        return self.preprocess_dict

    @classmethod
    def prepare(cls, runner: Runner):
        pass

    @classmethod
    def dispatch(cls, runner: Runner):
        data = runner.flow_data
        preprocess_object = cls.preprocess_dict[data.type]
        if not preprocess_object.match(data):
            raise RuntimeError('不支持的process')
        audio_np = preprocess_object(data)

        return audio_np

    @classmethod
    def support(cls, runner: Runner):
        for type_name in runner.processor_q:
            if cls.preprocess_dict[type_name] is None:
                raise NotImplementedError

    @classmethod
    def complete(cls, runner: Runner):
        pass

    @preprocess_dict.setter
    def preprocess_dict(self, value):
        self._preprocess_dict = value
