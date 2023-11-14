from typing import Optional, Union, Dict

from speakers.common.registry import registry
from speakers.processors import BaseProcessor, ProcessorData
from io import BytesIO
import logging
import numpy as np
import edge_tts
import asyncio
import nest_asyncio
import util
import librosa
import traceback

logger = logging.getLogger('edge_to_voice')


def set_edge_to_voice_logger(l):
    global logger
    logger = l


class EdgeProcessorData(ProcessorData):
    """
        :param text: 生成文本
        :param tts_speaker: 讲话人id
        :param rate: 语速
        :param volume: 语气轻重

    """
    """生成文本"""
    text: str
    """讲话人id"""
    tts_speaker: int
    """语速"""
    rate: str
    """语气轻重"""
    volume: str


    @property
    def type(self) -> str:
        """Type of the Message, used for serialization."""
        return "EDGE"


@registry.register_processor("edge_to_voice")
class EdgeToVoice(BaseProcessor):

    def __init__(self):
        super().__init__()
        nest_asyncio.apply()
        try:
            logger.info('Loading voices speakers role from edge_tts...')
            self._tts_speakers_list = asyncio.get_event_loop().run_until_complete(edge_tts.list_voices())  # noqa
        except Exception as e:
            logger.error(f'Error in loading voices from edge_tts:  {traceback.format_exc()}')
            self._tts_speakers_list = []

    def __call__(
            self,
            data: EdgeProcessorData
    ):

        if data.text is None:
            raise RuntimeError('Please provide TTS text.')

        if data.tts_speaker is None:
            raise RuntimeError('Please provide TTS text.')
        # 同步调用协程代码
        tts_np, tts_sr = asyncio.get_event_loop().run_until_complete( self._call_edge_tts(data=data))

        return tts_np, tts_sr

    @property
    def tts_speakers_list(self):
        return self._tts_speakers_list

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            raise RuntimeError("from_config cfg is None.")

        return cls()

    def match(self, data: ProcessorData):
        return "EDGE" in data.type

    async def _call_edge_tts(self, data: EdgeProcessorData):

        speaker = self._tts_speakers_list[data.tts_speaker]['ShortName']
        tts_com = edge_tts.Communicate(text=data.text, voice=speaker, rate=data.rate, volume=data.volume)
        tts_raw = b''

        # Stream TTS audio to bytes
        async for chunk in tts_com.stream():
            if chunk['type'] == 'audio':
                tts_raw += chunk['data']

        # Convert mp3 stream to wav
        ffmpeg_proc = await asyncio.create_subprocess_exec(
            'ffmpeg',
            '-f', 'mp3',
            '-i', '-',
            '-f', 'wav',
            '-loglevel', 'error',
            '-',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE
        )
        (tts_wav, _) = await ffmpeg_proc.communicate(tts_raw)

        return librosa.load(BytesIO(tts_wav))
