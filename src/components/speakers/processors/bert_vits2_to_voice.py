from speakers.vits import utils
from speakers.vits.modules import commons
from speakers.vits.models import SynthesizerTrn
from speakers.vits.text import text_to_sequence
from torch import no_grad, LongTensor
import torch
import os
from speakers.common.registry import registry
from speakers.processors import BaseProcessor, ProcessorData
import logging

logger = logging.getLogger('speaker_runner')


def set_bert_vits2_to_voice_logger(l):
    global logger
    logger = l


class BertVits2ProcessorData(ProcessorData):
    """

    """
    text: str
    slicer = gr.Button("快速切分", variant="primary")
    speaker = gr.Dropdown(
        choices=speakers, value=speakers[0], label="选择说话人"
    )
    sdp_ratio = gr.Slider(
        minimum=0, maximum=1, value=0.2, step=0.1, label="SDP/DP混合比"
    )
    noise_scale = gr.Slider(
        minimum=0.1, maximum=2, value=0.6, step=0.1, label="感情"
    )
    noise_scale_w = gr.Slider(
        minimum=0.1, maximum=2, value=0.8, step=0.1, label="音素长度"
    )
    length_scale = gr.Slider(
        minimum=0.1, maximum=2, value=1.0, step=0.1, label="语速"
    )
    language = gr.Dropdown(
        choices=languages, value=languages[0], label="选择语言(新增mix混合选项)"
    )

    @property
    def type(self) -> str:
        """Type of the Message, used for serialization."""
        return "VITS"


@registry.register_processor("vits_to_voice")
class VitsToVoice(BaseProcessor):

    def __init__(self, vits_model_path: str, voice_config_file: str):
        super().__init__()
        import nest_asyncio
        nest_asyncio.apply()
        self.limitation = os.getenv("SYSTEM") == "spaces"  # limit text and audio length in huggingface spaces
        logger.info(f'limit text and audio length in huggingface spaces: {self.limitation}')
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

        logger.info(f'_load_voice_mode: {voice_config_file}')
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
        logger.info(f'Models loaded vits_to_voice')

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
