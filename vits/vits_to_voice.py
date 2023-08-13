from vits import utils
from vits.modules import commons
from vits.models import SynthesizerTrn
from vits.text import text_to_sequence
from torch import no_grad, LongTensor
import torch
import os
from speakers.common.registry import registry

def get_text(text, hps):
    text_norm, clean_text = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm, clean_text


class VitsToVoice:

    def __init__(self, vits_model: str, voice_config_file: str):
        import nest_asyncio
        nest_asyncio.apply()
        self.limitation = os.getenv("SYSTEM") == "spaces"  # limit text and audio length in huggingface spaces
        self._load_voice_mode(vits_model=vits_model, voice_config_file=voice_config_file)

        self.language_marks = {
            "Japanese": "",
            "日本語": "[JA]",
            "简体中文": "[ZH]",
            "English": "[EN]",
            "Mix": "",
        }
        self.lang = ['日本語', '简体中文', 'English', 'Mix']

    def _load_voice_mode(self, vits_model: str, voice_config_file: str):

        device = torch.device(registry.get("device"))
        self.hps_ms = utils.get_hparams_from_file(voice_config_file)
        self.net_g_ms = SynthesizerTrn(
            len(self.hps_ms.symbols),
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            n_speakers=self.hps_ms.data.n_speakers, **self.hps_ms.model)

        _ = self.net_g_ms.eval().to(device)

        self.speakers = self.hps_ms.speakers
        self.model, self.optimizer, self.learning_rate, self.epochs = utils.load_checkpoint(vits_model, self.net_g_ms,
                                                                                            None)
        print(f'Models loaded')

    def search_speaker(self, search_value):
        """
        检索讲话人
        :return:
        """
        for s in self.speakers:
            if search_value == s:
                return s
        for s in self.speakers:
            if search_value in s:
                return s

    def vits_func(
            self,
            text: str, language: int, speaker_id: int,
            noise_scale=1, noise_scale_w=1, speed=1):
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
            text = self.language_marks[self.lang[language]] + text + self.language_marks[self.lang[language]]

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
