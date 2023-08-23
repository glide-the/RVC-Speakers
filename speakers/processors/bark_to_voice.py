# from vits import utils
# from vits.modules import commons
# from vits.models import SynthesizerTrn
# from vits.text import text_to_sequence
# from torch import no_grad, LongTensor
# import torch
# import os
# from speakers.common.registry import registry
# from speakers.processors import BaseProcessor, ProcessorData
# import logging
#
# logger = logging.getLogger('speaker_runner')
#
#
# def set_vits_to_voice_logger(l):
#     global logger
#     logger = l
#
#
# def get_text(text, hps):
#     text_norm, clean_text = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
#     if hps.data.add_blank:
#         text_norm = commons.intersperse(text_norm, 0)
#     text_norm = LongTensor(text_norm)
#     return text_norm, clean_text
#
#
# class BarkProcessorData(ProcessorData):
#     """
#         :param text: 生成文本
#         :param speaker_history_prompt: 音频预设npz文件
#         :param text_temp: 提示特殊标记程序，趋近于1，提示词特殊标记越明显
#         :param waveform_temp: 提示隐藏空间转音频参数比例
#
#     """
#     """生成文本"""
#     text: str
#     """音频预设npz文件"""
#     speaker_history_prompt: str
#     """提示特殊标记程序，趋近于1，提示词特殊标记越明显"""
#     text_temp: str
#     """提示隐藏空间转音频参数比例"""
#     waveform_temp: float
#
#     @property
#     def type(self) -> str:
#         """Type of the Message, used for serialization."""
#         return "BARK"
#
#
# class BarkToVoice(BaseProcessor):
#
#     model_conv: dict = {
#         "text"
#     }
#
#     def __init__(self, vits_model_path: str, voice_config_file: str):
#         super().__init__()
#         self._load_voice_mode(vits_model=vits_model_path, voice_config_file=voice_config_file)
#
#
#     def __call__(
#             self,
#             data: BarkProcessorData
#     ):
#
#     @classmethod
#     def from_config(cls, cfg=None):
#         if cfg is None:
#             raise RuntimeError("from_config cfg is None.")
#
#         vits_model_path = cfg.get("vits_model_path", "")
#         voice_config_file = cfg.get("voice_config_file", "")
#
#         return cls(vits_model_path=os.path.join(registry.get_path("vits_library_root"),
#                                                 vits_model_path),
#                    voice_config_file=os.path.join(registry.get_path("vits_library_root"),
#                                                   voice_config_file))
#
#     def match(self, data: ProcessorData):
#         return "VITS" in data.type
#
#
#     def _load__mode(self, vits_model: str, voice_config_file: str):
#
#         logger.info(f'_load_voice_mode: {voice_config_file}')
#
#         logger.info(f'Models loaded vits_to_voice')
#
#     def search_speaker(self, search_value):
#         """
#         检索讲话人
#         :return:
#         """
#         for s in self._speakers:
#             if search_value == s or search_value in s:
#                 return s
#
#     def vits_func(
#             self,
#             text: str, language: int, speaker_id: int,
#             noise_scale: float = 1, noise_scale_w: float = 1, speed=1):
#         """
#
#         :param text: 生成文本
#         :param language: 默认自动判断语言- 0中文， 1 日文
#         :param speaker_id: 讲话人id
#         :param noise_scale: noise_scale(控制感情变化程度)
#         :param speed: length_scale(控制整体语速)
#         :param noise_scale_w: noise_scale_w(控制音素发音长度)
#         :return:
#         """
#         if language is not None:
#             text = self._language_marks[self._lang[language]] + text + self._language_marks[self._lang[language]]
#
#         stn_tst, clean_text = get_text(text, self.hps_ms)
#         with no_grad():
#             x_tst = stn_tst.unsqueeze(0).to(registry.get("device"))
#             x_tst_lengths = LongTensor([stn_tst.size(0)]).to(registry.get("device"))
#             sid = LongTensor([speaker_id]).to(registry.get("device"))
#             audio = self.model.infer(x_tst, x_tst_lengths, sid=sid,
#                                      noise_scale=noise_scale,
#                                      noise_scale_w=noise_scale_w,
#                                      length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
#         del stn_tst, x_tst, x_tst_lengths, sid
#         return audio
