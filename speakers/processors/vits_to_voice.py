from vits import utils
from vits.modules import commons
from vits.models import SynthesizerTrn
from vits.text import text_to_sequence
from torch import no_grad, LongTensor
import torch
import os
from speakers.common.registry import registry
from speakers.processors import BaseProcessor, ProcessorData
from speakers.common.utils import get_abs_path
from omegaconf import OmegaConf
from pydantic import Field


def get_text(text, hps):
    text_norm, clean_text = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm, clean_text


class VitsProcessorData(ProcessorData):
    """
        :param text: 生成文本
        :param language: 默认自动判断语言- 0中文， 1 日文
        :param speaker_id: 讲话人id
        :param noise_scale: noise_scale(控制感情变化程度)
        :param speed: length_scale(控制整体语速)
        :param noise_scale_w: noise_scale_w(控制音素发音长度)

    """
    """生成文本"""
    text: str
    """语言- 0中文， 1 日文"""
    language: int
    """讲话人id"""
    speaker_id: int
    """ noise_scale(控制感情变化程度)"""
    noise_scale: float
    """length_scale(控制整体语速)"""
    speed: int
    """noise_scale_w(控制音素发音长度)"""
    noise_scale_w: float

    @property
    def type(self) -> str:
        """Type of the Message, used for serialization."""
        return "VITS"


@registry.register_processor("vits_to_voice")
class VitsToVoice(BaseProcessor):

    def __init__(self, vits_model_path: str, voice_config_file: str):
        import nest_asyncio
        nest_asyncio.apply()
        self.limitation = os.getenv("SYSTEM") == "spaces"  # limit text and audio length in huggingface spaces
        self._load_voice_mode(vits_model=vits_model_path, voice_config_file=voice_config_file)

        self._language_marks = {
            "Japanese": "",
            "日本語": "[JA]",
            "简体中文": "[ZH]",
            "English": "[EN]",
            "Mix": "",
        }
        self._lang = ['日本語', '简体中文', 'English', 'Mix']

    def __call__(
            self,
            data: VitsProcessorData
    ):

        return self.vits_func(text=data.text,
                              language=data.language,
                              speaker_id=data.speaker_id,
                              noise_scale=data.noise_scale,
                              noise_scale_w=data.noise_scale_w,
                              speed=data.speed)


    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            raise RuntimeError("from_config cfg is None.")

        vits_model_path = cfg.get("vits_model_path", "")
        voice_config_file = cfg.get("voice_config_file", "")

        return cls(vits_model_path=os.path.join(registry.get_path("vits_library_root"),
                                                vits_model_path),
                   voice_config_file=os.path.join(registry.get_path("vits_library_root"),
                                                  voice_config_file))

    def match(self, data: ProcessorData):
        return "VITS" in data.type

    @property
    def speakers(self):
        return self._speakers

    @property
    def lang(self):
        return self._lang

    @property
    def language_marks(self):
        return self._language_marks

    def _load_voice_mode(self, vits_model: str, voice_config_file: str):

        device = torch.device(registry.get("device"))
        self.hps_ms = utils.get_hparams_from_file(voice_config_file)
        self.net_g_ms = SynthesizerTrn(
            len(self.hps_ms.symbols),
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            n_speakers=self.hps_ms.data.n_speakers, **self.hps_ms.model)

        _ = self.net_g_ms.eval().to(device)

        self._speakers = self.hps_ms.speakers
        self.model, self.optimizer, self.learning_rate, self.epochs = utils.load_checkpoint(vits_model, self.net_g_ms,
                                                                                            None)
        print(f'Models loaded')

    def search_speaker(self, search_value):
        """
        检索讲话人
        :return:
        """
        for s in self._speakers:
            if search_value == s or search_value in s:
                return s

    def vits_func(
            self,
            text: str, language: int, speaker_id: int,
            noise_scale: float = 1, noise_scale_w: float = 1, speed=1):
        """

        :param text: 生成文本
        :param language: 默认自动判断语言- 0中文， 1 日文
        :param speaker_id: 讲话人id
        :param noise_scale: noise_scale(控制感情变化程度)
        :param speed: length_scale(控制整体语速)
        :param noise_scale_w: noise_scale_w(控制音素发音长度)
        :return:
        """
        if language is not None:
            text = self._language_marks[self._lang[language]] + text + self._language_marks[self._lang[language]]

        stn_tst, clean_text = get_text(text, self.hps_ms)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(registry.get("device"))
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(registry.get("device"))
            sid = LongTensor([speaker_id]).to(registry.get("device"))
            audio = self.model.infer(x_tst, x_tst_lengths, sid=sid,
                                     noise_scale=noise_scale,
                                     noise_scale_w=noise_scale_w,
                                     length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return audio
