from pathlib import Path

from speakers.bark.model_fine import FineGPT, FineGPTConfig
from speakers.bark.model import GPT, GPTConfig
from huggingface_hub import hf_hub_download
from typing import Type, TypeVar
from transformers import BertTokenizer
from scipy.special import softmax
from encodec import EncodecModel
import logging
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np
import contextlib
import funcy
import tqdm
import numpy as np
logger = logging.getLogger('bark_model_load')


def set_bark_model_load_logger(l):
    global logger
    logger = l



if (
        torch.cuda.is_available() and
        hasattr(torch.cuda, "amp") and
        hasattr(torch.cuda.amp, "autocast") and
        hasattr(torch.cuda, "is_bf16_supported") and
        torch.cuda.is_bf16_supported()
):
    autocast = funcy.partial(torch.cuda.amp.autocast, dtype=torch.bfloat16)
else:
    @contextlib.contextmanager
    def autocast():
        yield


class InferenceContext:
    def __init__(self, benchmark=False):
        # we can't expect inputs to be the same length, so disable benchmarking by default
        self._chosen_cudnn_benchmark = benchmark
        self._cudnn_benchmark = None

    def __enter__(self):
        self._cudnn_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = self._chosen_cudnn_benchmark

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.backends.cudnn.benchmark = self._cudnn_benchmark


@contextlib.contextmanager
def _inference_mode():
    with InferenceContext(), torch.inference_mode(), torch.no_grad(), autocast():
        yield


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


CONTEXT_WINDOW_SIZE = 1024

SEMANTIC_RATE_HZ = 49.9
SEMANTIC_VOCAB_SIZE = 10_000

CODEBOOK_SIZE = 1024
N_COARSE_CODEBOOKS = 2
N_FINE_CODEBOOKS = 8
COARSE_RATE_HZ = 75

SAMPLE_RATE = 24_000

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
SUPPORTED_LANGS = [
    ("English", "en"),
    ("German", "de"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("Hindi", "hi"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Turkish", "tr"),
    ("Chinese", "zh"),
]
# init prompt_speaker_np
ALLOWED_PROMPTS = {"announcer"}
for _, lang in SUPPORTED_LANGS:
    for prefix in ("", f"v2{os.path.sep}"):
        for n in range(10):
            ALLOWED_PROMPTS.add(f"{prefix}{lang}_speaker_{n}")

TEXT_ENCODING_OFFSET = 10_048
SEMANTIC_PAD_TOKEN = 10_000
TEXT_PAD_TOKEN = 129_595
SEMANTIC_INFER_TOKEN = 129_599

COARSE_SEMANTIC_PAD_TOKEN = 12_048
COARSE_INFER_TOKEN = 12_050


def _download(self, from_hf_path, file_name, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    hf_hub_download(repo_id=from_hf_path, filename=file_name, local_dir=local_dir)


def _load_codec_model(device,codec_repository_path: str):
    model = EncodecModel.encodec_model_24khz(pretrained=True, repository=Path(codec_repository_path))
    model.set_target_bandwidth(6.0)
    model.eval()
    model.to(device)
    return model


def _normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def _load_history_prompt(history_prompt_dir: str, history_prompt_input: str):
    # make sure this works on non-ubuntu

    history_prompt_input = os.path.join(*history_prompt_input.split("/"))
    if history_prompt_input not in ALLOWED_PROMPTS:
        raise ValueError("history prompt not found")
    history_prompt = np.load(
        os.path.join(history_prompt_dir, "assets", "prompts", f"{history_prompt_input}.npz")
    )
    return history_prompt


def _flatten_codebooks(arr, offset_size=CODEBOOK_SIZE):
    assert len(arr.shape) == 2
    arr = arr.copy()
    if offset_size is not None:
        for n in range(1, arr.shape[0]):
            arr[n, :] += offset_size * n
    flat_arr = arr.ravel("F")
    return flat_arr


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
    _text_model: ModelCheckPointInfo = ModelCheckPointInfo(
        model_type=ModelType(model_type="text_model", model_class=GPT, model_config=GPTConfig))
    _coarse_model: ModelCheckPointInfo = ModelCheckPointInfo(
        model_type=ModelType(model_type="coarse_model", model_class=GPT, model_config=GPTConfig))
    _fine_model: ModelCheckPointInfo = ModelCheckPointInfo(
        model_type=ModelType(model_type="fine_model", model_class=FineGPT, model_config=FineGPTConfig))

    _tokenizer: BertTokenizer
    _tokenizer_path: str = "bert-base-multilingual-cased"
    _encodec: EncodecModel

    def __init__(self, codec_repository_path: str, tokenizer_path: str, text_path: str, coarse_path: str, fine_path: str, device: str):

        if tokenizer_path:
            self._tokenizer_path = tokenizer_path
        logger.info(f"BertTokenizer load.")
        self._tokenizer = BertTokenizer.from_pretrained(self._tokenizer_path)
        logger.info(f"BertTokenizer loaded")

        logger.info(f"_encodec load.")
        self._encodec = _load_codec_model(device=device, codec_repository_path=codec_repository_path)
        logger.info(f"_encodec loaded")

        self._text_model.model_path = text_path
        self._coarse_model.model_path = coarse_path
        self._fine_model.model_path = fine_path

        # if not os.path.exists(self._text_model.model_path):
        #     model_info = REMOTE_MODEL_PATHS['text']
        #     logger.info(f"text model not found, downloading into `{self._text_model.model_path}`.")
        #
        #     _download(model_info["repo_id"], model_info["file_name"], self._text_model.model_path)
        # if not os.path.exists(self._coarse_model.model_path):
        #     model_info = REMOTE_MODEL_PATHS['coarse']
        #     logger.info(f"coarse model not found, downloading into `{self._coarse_model.model_path}`.")
        #     _download(model_info["repo_id"], model_info["file_name"], self._coarse_model.model_path)
        # if not os.path.exists(self._fine_model.model_path):
        #     model_info = REMOTE_MODEL_PATHS['fine']
        #     logger.info(f"fine model not found, downloading into `{self._fine_model.model_path}`.")
        #     _download(model_info["repo_id"], model_info["file_name"], self._fine_model.model_path)

        self._load_moad(model_type=self._text_model.model_type, ckpt_path=self._text_model.model_path, device=device)
        self._load_moad(model_type=self._coarse_model.model_type, ckpt_path=self._coarse_model.model_path, device=device)
        self._load_moad(model_type=self._fine_model.model_type, ckpt_path=self._fine_model.model_path, device=device)

    def _load_moad(self, model_type: ModelType, ckpt_path: str, device: str):
        if not os.path.exists(self._fine_model.model_path):
            raise RuntimeError("loader model path is not exists")

        device = torch.device(device)
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
            self._text_model.model = model
        elif model_type.model_type == "coarse_model":
            self._coarse_model.model = model
        elif model_type.model_type == "fine_model":
            self._fine_model.model = model

    def generate_text_semantic(
            self,
            text,
            history_prompt=None,
            history_prompt_dir=None,
            temp=0.7,
            top_k=None,
            top_p=None,
            silent=False,
            min_eos_p=0.2,
            max_gen_duration_s=None,
            allow_early_stop=True,
            use_kv_caching=False,
    ):
        """Generate semantic tokens from text."""
        assert isinstance(text, str)
        text = _normalize_whitespace(text)
        assert len(text.strip()) > 0
        if history_prompt is not None:
            history_prompt = _load_history_prompt(history_prompt_dir=history_prompt_dir, history_prompt_input=history_prompt)
            semantic_history = history_prompt["semantic_prompt"]
            assert (
                    isinstance(semantic_history, np.ndarray)
                    and len(semantic_history.shape) == 1
                    and len(semantic_history) > 0
                    and semantic_history.min() >= 0
                    and semantic_history.max() <= SEMANTIC_VOCAB_SIZE - 1
            )
        else:
            semantic_history = None
        model = self._text_model.model
        tokenizer = self._tokenizer
        encoded_text = np.array(tokenizer.encode(text, add_special_tokens=False)) + TEXT_ENCODING_OFFSET
        # if OFFLOAD_CPU:
        #     model.to(models_devices["text"])
        device = next(model.parameters()).device
        if len(encoded_text) > 256:
            p = round((len(encoded_text) - 256) / len(encoded_text) * 100, 1)
            logger.warning(f"warning, text too long, lopping of last {p}%")
            encoded_text = encoded_text[:256]
        encoded_text = np.pad(
            encoded_text,
            (0, 256 - len(encoded_text)),
            constant_values=TEXT_PAD_TOKEN,
            mode="constant",
        )
        if semantic_history is not None:
            semantic_history = semantic_history.astype(np.int64)
            # lop off if history is too long, pad if needed
            semantic_history = semantic_history[-256:]
            semantic_history = np.pad(
                semantic_history,
                (0, 256 - len(semantic_history)),
                constant_values=SEMANTIC_PAD_TOKEN,
                mode="constant",
            )
        else:
            semantic_history = np.array([SEMANTIC_PAD_TOKEN] * 256)
        x = torch.from_numpy(
            np.hstack([
                encoded_text, semantic_history, np.array([SEMANTIC_INFER_TOKEN])
            ]).astype(np.int64)
        )[None]
        assert x.shape[1] == 256 + 256 + 1
        with _inference_mode():
            x = x.to(device)
            n_tot_steps = 768
            # custom tqdm updates since we don't know when eos will occur
            pbar = tqdm.tqdm(disable=silent, total=n_tot_steps)
            pbar_state = 0
            tot_generated_duration_s = 0
            kv_cache = None
            for n in range(n_tot_steps):
                if use_kv_caching and kv_cache is not None:
                    x_input = x[:, [-1]]
                else:
                    x_input = x
                logits, kv_cache = model(
                    x_input, merge_context=True, use_cache=use_kv_caching, past_kv=kv_cache
                )
                relevant_logits = logits[0, 0, :SEMANTIC_VOCAB_SIZE]
                if allow_early_stop:
                    relevant_logits = torch.hstack(
                        (relevant_logits, logits[0, 0, [SEMANTIC_PAD_TOKEN]])  # eos
                    )
                if top_p is not None:
                    # faster to convert to numpy
                    original_device = relevant_logits.device
                    relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
                    sorted_indices = np.argsort(relevant_logits)[::-1]
                    sorted_logits = relevant_logits[sorted_indices]
                    cumulative_probs = np.cumsum(softmax(sorted_logits))
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                    sorted_indices_to_remove[0] = False
                    relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                    relevant_logits = torch.from_numpy(relevant_logits)
                    relevant_logits = relevant_logits.to(original_device)
                if top_k is not None:
                    v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                    relevant_logits[relevant_logits < v[-1]] = -float("Inf")
                probs = F.softmax(relevant_logits / temp, dim=-1)
                item_next = torch.multinomial(probs, num_samples=1).to(torch.int32)
                if allow_early_stop and (
                        item_next == SEMANTIC_VOCAB_SIZE
                        or (min_eos_p is not None and probs[-1] >= min_eos_p)
                ):
                    # eos found, so break
                    pbar.update(n - pbar_state)
                    break
                x = torch.cat((x, item_next[None]), dim=1)
                tot_generated_duration_s += 1 / SEMANTIC_RATE_HZ
                if max_gen_duration_s is not None and tot_generated_duration_s > max_gen_duration_s:
                    pbar.update(n - pbar_state)
                    break
                if n == n_tot_steps - 1:
                    pbar.update(n - pbar_state)
                    break
                del logits, relevant_logits, probs, item_next

                if n > pbar_state:
                    if n > pbar.total:
                        pbar.total = n
                    pbar.update(n - pbar_state)
                pbar_state = n
            pbar.total = n
            pbar.refresh()
            pbar.close()
            out = x.detach().cpu().numpy().squeeze()[256 + 256 + 1:]

        assert all(0 <= out) and all(out < SEMANTIC_VOCAB_SIZE)

        return out

    def generate_coarse(
            self,
            x_semantic,
            history_prompt=None,
            history_prompt_dir=None,
            temp=0.7,
            top_k=None,
            top_p=None,
            silent=False,
            max_coarse_history=630,  # min 60 (faster), max 630 (more context)
            sliding_window_len=60,
            use_kv_caching=False,
    ):
        """Generate coarse audio codes from semantic tokens."""
        assert (
                isinstance(x_semantic, np.ndarray)
                and len(x_semantic.shape) == 1
                and len(x_semantic) > 0
                and x_semantic.min() >= 0
                and x_semantic.max() <= SEMANTIC_VOCAB_SIZE - 1
        )
        assert 60 <= max_coarse_history <= 630
        assert max_coarse_history + sliding_window_len <= 1024 - 256
        semantic_to_coarse_ratio = COARSE_RATE_HZ / SEMANTIC_RATE_HZ * N_COARSE_CODEBOOKS
        max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))
        if history_prompt is not None:
            history_prompt = _load_history_prompt(history_prompt_dir=history_prompt_dir,history_prompt_input=history_prompt)
            x_semantic_history = history_prompt["semantic_prompt"]
            x_coarse_history = history_prompt["coarse_prompt"]
            assert (
                    isinstance(x_semantic_history, np.ndarray)
                    and len(x_semantic_history.shape) == 1
                    and len(x_semantic_history) > 0
                    and x_semantic_history.min() >= 0
                    and x_semantic_history.max() <= SEMANTIC_VOCAB_SIZE - 1
                    and isinstance(x_coarse_history, np.ndarray)
                    and len(x_coarse_history.shape) == 2
                    and x_coarse_history.shape[0] == N_COARSE_CODEBOOKS
                    and x_coarse_history.shape[-1] >= 0
                    and x_coarse_history.min() >= 0
                    and x_coarse_history.max() <= CODEBOOK_SIZE - 1
                    and (
                            round(x_coarse_history.shape[-1] / len(x_semantic_history), 1)
                            == round(semantic_to_coarse_ratio / N_COARSE_CODEBOOKS, 1)
                    )
            )
            x_coarse_history = _flatten_codebooks(x_coarse_history) + SEMANTIC_VOCAB_SIZE
            # trim histories correctly
            n_semantic_hist_provided = np.min(
                [
                    max_semantic_history,
                    len(x_semantic_history) - len(x_semantic_history) % 2,
                    int(np.floor(len(x_coarse_history) / semantic_to_coarse_ratio)),
                ]
            )
            n_coarse_hist_provided = int(round(n_semantic_hist_provided * semantic_to_coarse_ratio))
            x_semantic_history = x_semantic_history[-n_semantic_hist_provided:].astype(np.int32)
            x_coarse_history = x_coarse_history[-n_coarse_hist_provided:].astype(np.int32)
            # TODO: bit of a hack for time alignment (sounds better)
            x_coarse_history = x_coarse_history[:-2]
        else:
            x_semantic_history = np.array([], dtype=np.int32)
            x_coarse_history = np.array([], dtype=np.int32)

        model = self._coarse_model.model
        # if OFFLOAD_CPU:
        #     model.to(models_devices["coarse"])
        device = next(model.parameters()).device
        # start loop
        n_steps = int(
            round(
                np.floor(len(x_semantic) * semantic_to_coarse_ratio / N_COARSE_CODEBOOKS)
                * N_COARSE_CODEBOOKS
            )
        )
        assert n_steps > 0 and n_steps % N_COARSE_CODEBOOKS == 0
        x_semantic = np.hstack([x_semantic_history, x_semantic]).astype(np.int32)
        x_coarse = x_coarse_history.astype(np.int32)
        base_semantic_idx = len(x_semantic_history)
        with _inference_mode():
            x_semantic_in = torch.from_numpy(x_semantic)[None].to(device)
            x_coarse_in = torch.from_numpy(x_coarse)[None].to(device)
            n_window_steps = int(np.ceil(n_steps / sliding_window_len))
            n_step = 0
            for _ in tqdm.tqdm(range(n_window_steps), total=n_window_steps, disable=silent):
                semantic_idx = base_semantic_idx + int(round(n_step / semantic_to_coarse_ratio))
                # pad from right side
                x_in = x_semantic_in[:, np.max([0, semantic_idx - max_semantic_history]):]
                x_in = x_in[:, :256]
                x_in = F.pad(
                    x_in,
                    (0, 256 - x_in.shape[-1]),
                    "constant",
                    COARSE_SEMANTIC_PAD_TOKEN,
                )
                x_in = torch.hstack(
                    [
                        x_in,
                        torch.tensor([COARSE_INFER_TOKEN])[None].to(device),
                        x_coarse_in[:, -max_coarse_history:],
                    ]
                )
                kv_cache = None
                for _ in range(sliding_window_len):
                    if n_step >= n_steps:
                        continue
                    is_major_step = n_step % N_COARSE_CODEBOOKS == 0

                    if use_kv_caching and kv_cache is not None:
                        x_input = x_in[:, [-1]]
                    else:
                        x_input = x_in

                    logits, kv_cache = model(x_input, use_cache=use_kv_caching, past_kv=kv_cache)
                    logit_start_idx = (
                            SEMANTIC_VOCAB_SIZE + (1 - int(is_major_step)) * CODEBOOK_SIZE
                    )
                    logit_end_idx = (
                            SEMANTIC_VOCAB_SIZE + (2 - int(is_major_step)) * CODEBOOK_SIZE
                    )
                    relevant_logits = logits[0, 0, logit_start_idx:logit_end_idx]
                    if top_p is not None:
                        # faster to convert to numpy
                        original_device = relevant_logits.device
                        relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
                        sorted_indices = np.argsort(relevant_logits)[::-1]
                        sorted_logits = relevant_logits[sorted_indices]
                        cumulative_probs = np.cumsum(softmax(sorted_logits))
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                        sorted_indices_to_remove[0] = False
                        relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                        relevant_logits = torch.from_numpy(relevant_logits)
                        relevant_logits = relevant_logits.to(original_device)
                    if top_k is not None:
                        v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                        relevant_logits[relevant_logits < v[-1]] = -float("Inf")
                    probs = F.softmax(relevant_logits / temp, dim=-1)
                    item_next = torch.multinomial(probs, num_samples=1).to(torch.int32)
                    item_next += logit_start_idx
                    x_coarse_in = torch.cat((x_coarse_in, item_next[None]), dim=1)
                    x_in = torch.cat((x_in, item_next[None]), dim=1)
                    del logits, relevant_logits, probs, item_next
                    n_step += 1
                del x_in
            del x_semantic_in
        # if OFFLOAD_CPU:
        #     model.to("cpu")
        gen_coarse_arr = x_coarse_in.detach().cpu().numpy().squeeze()[len(x_coarse_history):]
        del x_coarse_in
        assert len(gen_coarse_arr) == n_steps
        gen_coarse_audio_arr = gen_coarse_arr.reshape(-1, N_COARSE_CODEBOOKS).T - SEMANTIC_VOCAB_SIZE
        for n in range(1, N_COARSE_CODEBOOKS):
            gen_coarse_audio_arr[n, :] -= n * CODEBOOK_SIZE
        # _clear_cuda_cache()
        return gen_coarse_audio_arr

    def generate_fine(
            self,
            x_coarse_gen,
            history_prompt=None,
            history_prompt_dir=None,
            temp=0.5,
            silent=True,
    ):
        """Generate full audio codes from coarse audio codes."""
        assert (
                isinstance(x_coarse_gen, np.ndarray)
                and len(x_coarse_gen.shape) == 2
                and 1 <= x_coarse_gen.shape[0] <= N_FINE_CODEBOOKS - 1
                and x_coarse_gen.shape[1] > 0
                and x_coarse_gen.min() >= 0
                and x_coarse_gen.max() <= CODEBOOK_SIZE - 1
        )
        if history_prompt is not None:
            history_prompt = _load_history_prompt(history_prompt_dir=history_prompt_dir,history_prompt_input=history_prompt)
            x_fine_history = history_prompt["fine_prompt"]
            assert (
                    isinstance(x_fine_history, np.ndarray)
                    and len(x_fine_history.shape) == 2
                    and x_fine_history.shape[0] == N_FINE_CODEBOOKS
                    and x_fine_history.shape[1] >= 0
                    and x_fine_history.min() >= 0
                    and x_fine_history.max() <= CODEBOOK_SIZE - 1
            )
        else:
            x_fine_history = None
        n_coarse = x_coarse_gen.shape[0]

        model = self._fine_model.model
        # if OFFLOAD_CPU:
        #     model.to(models_devices["fine"])
        device = next(model.parameters()).device
        # make input arr
        in_arr = np.vstack(
            [
                x_coarse_gen,
                np.zeros((N_FINE_CODEBOOKS - n_coarse, x_coarse_gen.shape[1]))
                + CODEBOOK_SIZE,  # padding
            ]
        ).astype(np.int32)
        # prepend history if available (max 512)
        if x_fine_history is not None:
            x_fine_history = x_fine_history.astype(np.int32)
            in_arr = np.hstack(
                [
                    x_fine_history[:, -512:].astype(np.int32),
                    in_arr,
                ]
            )
            n_history = x_fine_history[:, -512:].shape[1]
        else:
            n_history = 0
        n_remove_from_end = 0
        # need to pad if too short (since non-causal model)
        if in_arr.shape[1] < 1024:
            n_remove_from_end = 1024 - in_arr.shape[1]
            in_arr = np.hstack(
                [
                    in_arr,
                    np.zeros((N_FINE_CODEBOOKS, n_remove_from_end), dtype=np.int32) + CODEBOOK_SIZE,
                ]
            )
        # we can be lazy about fractional loop and just keep overwriting codebooks
        n_loops = np.max([0, int(np.ceil((x_coarse_gen.shape[1] - (1024 - n_history)) / 512))]) + 1
        with _inference_mode():
            in_arr = torch.tensor(in_arr.T).to(device)
            for n in tqdm.tqdm(range(n_loops), disable=silent):
                start_idx = np.min([n * 512, in_arr.shape[0] - 1024])
                start_fill_idx = np.min([n_history + n * 512, in_arr.shape[0] - 512])
                rel_start_fill_idx = start_fill_idx - start_idx
                in_buffer = in_arr[start_idx: start_idx + 1024, :][None]
                for nn in range(n_coarse, N_FINE_CODEBOOKS):
                    logits = model(nn, in_buffer)
                    if temp is None:
                        relevant_logits = logits[0, rel_start_fill_idx:, :CODEBOOK_SIZE]
                        codebook_preds = torch.argmax(relevant_logits, -1)
                    else:
                        relevant_logits = logits[0, :, :CODEBOOK_SIZE] / temp
                        probs = F.softmax(relevant_logits, dim=-1)
                        codebook_preds = torch.multinomial(
                            probs[rel_start_fill_idx:1024], num_samples=1
                        ).reshape(-1)
                    codebook_preds = codebook_preds.to(torch.int32)
                    in_buffer[0, rel_start_fill_idx:, nn] = codebook_preds
                    del logits, codebook_preds
                # transfer over info into model_in and convert to numpy
                for nn in range(n_coarse, N_FINE_CODEBOOKS):
                    in_arr[
                    start_fill_idx: start_fill_idx + (1024 - rel_start_fill_idx), nn
                    ] = in_buffer[0, rel_start_fill_idx:, nn]
                del in_buffer
            gen_fine_arr = in_arr.detach().cpu().numpy().squeeze().T
            del in_arr
        # if OFFLOAD_CPU:
        #     model.to("cpu")
        gen_fine_arr = gen_fine_arr[:, n_history:]
        if n_remove_from_end > 0:
            gen_fine_arr = gen_fine_arr[:, :-n_remove_from_end]
        assert gen_fine_arr.shape[-1] == x_coarse_gen.shape[-1]
        # _clear_cuda_cache()
        return gen_fine_arr

    def codec_decode(self, fine_tokens):
        """Turn quantized audio codes into audio array using encodec."""

        model = self._encodec
        device = next(model.parameters()).device
        arr = torch.from_numpy(fine_tokens)[None]
        arr = arr.to(device)
        arr = arr.transpose(0, 1)
        emb = model.quantizer.decode(arr)
        out = model.decoder(emb)
        audio_arr = out.detach().cpu().numpy().squeeze()
        del arr, emb, out

        return audio_arr


if __name__ == '__main__':
    bark_load = BarkModelLoader(tokenizer_path='/media/checkpoint/bark/bert-base-multilingual-cased/',
                                text_path='/media/checkpoint/bark/suno/bark_v0/text_2.pt',
                                coarse_path='/media/checkpoint/bark/suno/bark_v0/coarse_2.pt',
                                fine_path='/media/checkpoint/bark/suno/bark_v0/fine_2.pt')

    print(bark_load)
