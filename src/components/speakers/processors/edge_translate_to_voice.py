from typing import Optional, Union, Dict, List

from speakers.common.registry import registry
from speakers.processors import BaseProcessor, ProcessorData
import logging
import json
import requests
import asyncio
import nest_asyncio

logger = logging.getLogger('edge_translate_to_voice')


def set_edge_translate_to_voice_logger(l):
    global logger
    logger = l


class CaiyunTranslateProcessorData(ProcessorData):
    """
        :param text: 文本
        :param direction: 目标语言

    """
    """文本"""
    source: List[str]
    """目标语言"""
    direction: str

    @property
    def type(self) -> str:
        """Type of the Message, used for serialization."""
        return "EDGE_TRANSLATE"


@registry.register_processor("caiyun_translate_text")
class CaiyunTranslateText(BaseProcessor):
    _url: str = "http://api.interpreter.caiyunai.com/v1/translator"
    _token: str = None

    def __init__(self, token: str):
        super().__init__()
        self._token = token

    def __call__(
            self,
            data: CaiyunTranslateProcessorData
    ):
        # 同步调用协程代码
        translator_text = (
            asyncio.get_event_loop()
            .run_until_complete(self._translator(source=data.source,direction=data.direction))
        )

        return translator_text

    @property
    def token(self):
        return self._token

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            raise RuntimeError("from_config cfg is None.")

        token = cfg.get("token", "")
        return cls(token=token)

    def match(self, data: ProcessorData):
        return "EDGE_TRANSLATE" in data.type

    async def _translator(self, source: List[str], direction: str) -> List[str]:
        payload = {
            "source": source,
            "trans_type": direction,
            "request_id": "demo",
            "detect": True,
        }

        headers = {
            "content-type": "application/json",
            "x-authorization": "token " + self._token,
        }

        response = requests.request("POST", self._url, data=json.dumps(payload), headers=headers)

        return json.loads(response.text)["target"]
