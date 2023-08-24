from typing import Optional, Union, Dict
from bark.mode_load import BarkModelLoader, SAMPLE_RATE
from vits.modules import commons
from vits.text import text_to_sequence
from torch import LongTensor
from speakers.common.registry import registry
from speakers.processors import BaseProcessor, ProcessorData
import os
import logging
import numpy as np
import nltk  # we'll use this to split into sentences
from nltk.tokenize import RegexpTokenizer

logger = logging.getLogger('bark_to_voice')


def set_bark_to_voice_logger(l):
    global logger
    logger = l


silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence


def get_text(text, hps):
    text_norm, clean_text = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm, clean_text


class BarkProcessorData(ProcessorData):
    """
        :param text: 生成文本
        :param speaker_history_prompt: 音频预设npz文件
        :param text_temp: 提示特殊标记程序，趋近于1，提示词特殊标记越明显
        :param waveform_temp: 提示隐藏空间转音频参数比例

    """
    """生成文本"""
    text: str
    """音频预设npz文件"""
    speaker_history_prompt: str
    """提示特殊标记程序，趋近于1，提示词特殊标记越明显"""
    text_temp: float
    """提示隐藏空间转音频参数比例"""
    waveform_temp: float

    @property
    def type(self) -> str:
        """Type of the Message, used for serialization."""
        return "BARK"


@registry.register_processor("bark_to_voice")
class BarkToVoice(BaseProcessor):

    def __init__(self,codec_repository_path: str, tokenizer_path: str, text_path: str, coarse_path: str, fine_path: str):
        super().__init__()
        self._load_bark_mode(codec_repository_path=codec_repository_path,
                             tokenizer_path=tokenizer_path,
                             text_path=text_path,
                             coarse_path=coarse_path,
                             fine_path=fine_path)

    def __call__(
            self,
            data: BarkProcessorData
    ):
        # 分词，适配长句子
        script = data.text.replace("\n", "。").strip()
        tokenizer = RegexpTokenizer(r'[^，。！？]+[，。！？]?')
        sentences = tokenizer.tokenize(script)

        pieces = []
        logger.info(f"sentences:{sentences}")
        for sentence in sentences:
            audio_array = self._generate_audio(text=sentence,
                                               history_prompt_dir=registry.get_path('bark_library_root'),
                                               history_prompt=data.speaker_history_prompt,
                                               text_temp=data.text_temp,
                                               waveform_temp=data.waveform_temp)

            pieces += [audio_array, silence.copy()]

        audio_array_out = np.concatenate(pieces)
        del pieces
        return audio_array_out

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            raise RuntimeError("from_config cfg is None.")

        codec_repository_path = cfg.get("codec_repository_path", "")
        tokenizer_path = cfg.get("tokenizer_path", "")
        text_model_path = cfg.get("text_model_path", "")
        coarse_model_path = cfg.get("coarse_model_path", "")
        fine_model_path = cfg.get("fine_model_path", "")

        return cls(codec_repository_path=os.path.join(registry.get_path("bark_library_root"),
                                                     codec_repository_path),
                   tokenizer_path=os.path.join(registry.get_path("bark_library_root"),
                                               tokenizer_path),
                   text_path=os.path.join(registry.get_path("bark_library_root"),
                                          text_model_path),
                   coarse_path=os.path.join(registry.get_path("bark_library_root"),
                                            coarse_model_path),
                   fine_path=os.path.join(registry.get_path("bark_library_root"),
                                          fine_model_path)
                   )

    def match(self, data: ProcessorData):
        return "BARK" in data.type

    def _load_bark_mode(self, codec_repository_path: str, tokenizer_path: str, text_path: str, coarse_path: str, fine_path: str):

        logger.info(f'Bark model loading')
        self.bark_load = BarkModelLoader(codec_repository_path=codec_repository_path,
                                         tokenizer_path=tokenizer_path,
                                         text_path=text_path,
                                         coarse_path=coarse_path,
                                         fine_path=fine_path,
                                         device=registry.get("device"))
        logger.info(f'Models loaded bark')

    def _generate_audio(
            self,
            text: str,
            history_prompt: Optional[str] = None,
            history_prompt_dir: str = None,
            text_temp: float = 0.7,
            waveform_temp: float = 0.7,
            fine_temp: float = 0.5,
            silent: bool = False,
            output_full: bool = False):
        """Generate audio array from input text.

        Args:
            text: text to be turned into audio
            history_prompt: history choice for audio cloning
            text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
            waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
            fine_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
            silent: disable progress bar
            output_full: return full generation to be used as a history prompt

        Returns:
            numpy audio array at sample frequency 24khz
        """
        semantic_tokens = self._text_to_semantic(
            text,
            history_prompt=history_prompt,
            history_prompt_dir=history_prompt_dir,
            temp=text_temp,
            silent=silent,
        )
        out = self._semantic_to_waveform(
            semantic_tokens,
            history_prompt=history_prompt,
            history_prompt_dir=history_prompt_dir,
            temp=waveform_temp,
            fine_temp=fine_temp,
            silent=silent,
            output_full=output_full,
        )
        if output_full:
            full_generation, audio_arr = out
            return full_generation, audio_arr
        else:
            audio_arr = out
        return audio_arr

    def _text_to_semantic(
            self,
            text: str,
            history_prompt: Optional[str] = None,
            history_prompt_dir=None,
            temp: float = 0.7,
            silent: bool = False,
    ):
        """Generate semantic array from text.

        Args:
            text: text to be turned into audio
            history_prompt: history choice for audio cloning
            temp: generation temperature (1.0 more diverse, 0.0 more conservative)
            silent: disable progress bar

        Returns:
            numpy semantic array to be fed into `semantic_to_waveform`
        """
        x_semantic = self.bark_load.generate_text_semantic(
            text,
            history_prompt=history_prompt,
            history_prompt_dir=history_prompt_dir,
            temp=temp,
            silent=silent,
            use_kv_caching=True
        )
        return x_semantic

    def _semantic_to_waveform(
            self,
            semantic_tokens: np.ndarray,
            history_prompt: Optional[Union[Dict, str]] = None,
            history_prompt_dir: str = None,
            temp: float = 0.7,
            fine_temp: float = 0.5,
            silent: bool = False,
            output_full: bool = False,
    ):
        """Generate audio array from semantic input.

        Args:
            semantic_tokens: semantic token output from `text_to_semantic`
            history_prompt: history choice for audio cloning
            fine_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
            temp: generation temperature (1.0 more diverse, 0.0 more conservative)
            silent: disable progress bar
            output_full: return full generation to be used as a history prompt

        Returns:
            numpy audio array at sample frequency 24khz
        """
        coarse_tokens = self.bark_load.generate_coarse(
            semantic_tokens,
            history_prompt=history_prompt,
            history_prompt_dir=history_prompt_dir,
            temp=temp,
            silent=silent,
            use_kv_caching=True
        )
        fine_tokens = self.bark_load.generate_fine(
            coarse_tokens,
            history_prompt=history_prompt,
            history_prompt_dir=history_prompt_dir,
            temp=fine_temp,
        )
        audio_arr = self.bark_load.codec_decode(fine_tokens)
        if output_full:
            full_generation = {
                "semantic_prompt": semantic_tokens,
                "coarse_prompt": coarse_tokens,
                "fine_prompt": fine_tokens,
            }
            return full_generation, audio_arr
        return audio_arr
