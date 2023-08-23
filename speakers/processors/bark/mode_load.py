from bark.model_fine import FineGPT, FineGPTConfig

from bark.model import GPT, GPTConfig
from huggingface_hub import hf_hub_download
from speakers.common.registry import registry
from typing import Type, TypeVar
from transformers import BertTokenizer
import logging
import torch
import os
import torch.nn as nn

logger = logging.getLogger('speaker_runner')


def set_rvc_speakers_logger(l):
    global logger
    logger = l


# 定义一个泛型类型变量
G = TypeVar('G', bound='GPT')
C = TypeVar('C', bound='GPTConfig')


class ModelType:
    def __init__(self, model_type: str, model_class: Type[G], model_config: Type[C]):
        """
        模型类型适配
        :param model_type:
        :param model_class:
        :param model_config:
        """
        self.model_type = model_type
        self.model_class = model_class
        self.model_config = model_config


REMOTE_MODEL_PATHS = {
    "text": {
        "repo_id": "suno/bark",
        "file_name": "text_2.pt",
    },
    "coarse": {
        "repo_id": "suno/bark",
        "file_name": "coarse_2.pt",
    },
    "fine": {
        "repo_id": "suno/bark",
        "file_name": "fine_2.pt",
    },
}


def _download(self, from_hf_path, file_name, local_dir):
    os.makedirs(CACHE_DIR, exist_ok=True)
    hf_hub_download(repo_id=from_hf_path, filename=file_name, local_dir=local_dir)


class ModelCheckPointInfo:
    _model_type: ModelType
    _model: nn.Module
    _model_path: str

    def __init__(self, model_type: ModelType):
        self._model_type = model_type

    @property
    def model_type(self) -> ModelType:
        return self._model_type

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def model_path(self) -> str:
        return self._model_path

    @model.setter
    def model(self, value):
        self._model = value

    @model_path.setter
    def model_path(self, value):
        self._model_path = value


class BarkModelLoader:
    text_model: ModelCheckPointInfo = ModelCheckPointInfo(
        model_type=ModelType(model_type="text_model", model_class=GPT, model_config=GPTConfig))
    coarse_model: ModelCheckPointInfo = ModelCheckPointInfo(
        model_type=ModelType(model_type="coarse_model", model_class=GPT, model_config=GPTConfig))
    fine_model: ModelCheckPointInfo = ModelCheckPointInfo(
        model_type=ModelType(model_type="fine_model", model_class=FineGPT, model_config=FineGPTConfig))

    _tokenizer: BertTokenizer
    _tokenizer_path: str = "bert-base-multilingual-cased"

    def __init__(self, tokenizer_path: str, text_path: str, coarse_path: str, fine_path: str):

        if tokenizer_path:
            self._tokenizer_path = tokenizer_path
        logger.info(f"BertTokenizer load.")
        self._tokenizer = BertTokenizer.from_pretrained(self._tokenizer_path)
        logger.info(f"BertTokenizer loaded")

        self.text_model.model_path = text_path
        self.coarse_model.model_path = coarse_path
        self.fine_model.model_path = fine_path

        # if not os.path.exists(self.text_model.model_path):
        #     model_info = REMOTE_MODEL_PATHS['text']
        #     logger.info(f"text model not found, downloading into `{self.text_model.model_path}`.")
        #
        #     _download(model_info["repo_id"], model_info["file_name"], self.text_model.model_path)
        # if not os.path.exists(self.coarse_model.model_path):
        #     model_info = REMOTE_MODEL_PATHS['coarse']
        #     logger.info(f"coarse model not found, downloading into `{self.coarse_model.model_path}`.")
        #     _download(model_info["repo_id"], model_info["file_name"], self.coarse_model.model_path)
        # if not os.path.exists(self.fine_model.model_path):
        #     model_info = REMOTE_MODEL_PATHS['fine']
        #     logger.info(f"fine model not found, downloading into `{self.fine_model.model_path}`.")
        #     _download(model_info["repo_id"], model_info["file_name"], self.fine_model.model_path)

        self._load_moad(model_type=self.text_model.model_type, ckpt_path=self.text_model.model_path)
        self._load_moad(model_type=self.coarse_model.model_type, ckpt_path=self.coarse_model.model_path)
        self._load_moad(model_type=self.fine_model.model_type, ckpt_path=self.fine_model.model_path)

    def _load_moad(self, model_type: ModelType, ckpt_path: str):
        if not os.path.exists(self.fine_model.model_path):
            raise RuntimeError("loader model path is not exists")

        device = torch.device(registry.get("device"))
        checkpoint = torch.load(ckpt_path, map_location=device)
        # this is a hack
        model_args = checkpoint["model_args"]
        if "input_vocab_size" not in model_args:
            model_args["input_vocab_size"] = model_args["vocab_size"]
            model_args["output_vocab_size"] = model_args["vocab_size"]
            del model_args["vocab_size"]
        gptconf = model_type.model_config(**checkpoint["model_args"])
        model = model_type.model_class(gptconf)
        state_dict = checkpoint["model"]
        # fixup checkpoint
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        extra_keys = set(state_dict.keys()) - set(model.state_dict().keys())
        extra_keys = set([k for k in extra_keys if not k.endswith(".attn.bias")])
        missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
        missing_keys = set([k for k in missing_keys if not k.endswith(".attn.bias")])
        if len(extra_keys) != 0:
            raise ValueError(f"extra keys found: {extra_keys}")
        if len(missing_keys) != 0:
            raise ValueError(f"missing keys: {missing_keys}")
        model.load_state_dict(state_dict, strict=False)
        n_params = model.get_num_params()
        val_loss = checkpoint["best_val_loss"].item()
        logger.info(
            f"model {model_type.model_type} loaded: {round(n_params / 1e6, 1)}M params, {round(val_loss, 3)} loss")
        model.eval()
        model.to(device)
        del checkpoint, state_dict
        if model_type.model_type == "text_model":
            self.text_model.model = model
        elif model_type.model_type == "coarse_model":
            self.coarse_model.model = model
        elif model_type.model_type == "fine_model":
            self.fine_model.model = model


if __name__ == '__main__':
    bark_load = BarkModelLoader(tokenizer_path='/media/checkpoint/bark/bert-base-multilingual-cased/',
                                text_path='/media/checkpoint/bark/suno/bark_v0/text_2.pt',
                                coarse_path='/media/checkpoint/bark/suno/bark_v0/coarse_2.pt',
                                fine_path='/media/checkpoint/bark/suno/bark_v0/fine_2.pt')

    print(bark_load)
