from speakers.common import util
import numpy as np
import librosa
import hashlib
import json
import os
import torch
import logging
from speakers.rvc.infer_pack.models import (
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono
)
from os import getenv
from typing import Union, Tuple, List
from speakers.rvc.vc_infer_pipeline import VC
from speakers.processors import BaseProcessor, ProcessorData
from speakers.common.utils import get_abs_path
from omegaconf import OmegaConf
from speakers.common.registry import registry
from pydantic import Field

logger = logging.getLogger('speaker_runner')


def set_rvc_speakers_logger(l):
    global logger
    logger = l


class RvcProcessorData(ProcessorData):
    """
        # https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/blob/main/infer-web.py#L118  # noqa
    :param f0_up_key:  变调(整数, 半音数量, 升八度12降八度-12)
    :param input_audio:
    :param f0_file:  F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调
    :param protect: 保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果
    :param model_index:
    :param f0_method:
    :param index_rate: 检索特征占比
    :param filter_radius: >=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音
    :param rms_mix_rate: 输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络
    :param resample_sr: 后处理重采样至最终采样率，0为不进行重采样
    :return:
    """
    sample_rate: int = Field(
        default=0
    )
    audio_samples: List[float] = Field(
        default=[]
    )

    model_index: int

    """ 变调(整数, 半音数量, 升八度12降八度-12)"""
    f0_up_key: int

    """ F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调"""
    f0_method: str

    """检索特征占比"""
    index_rate: float
    """ >=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"""
    filter_radius: int
    """输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"""
    rms_mix_rate: float
    """后处理重采样至最终采样率，0为不进行重采样"""
    resample_sr: int
    """保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"""
    protect: float = Field(
        default=0.33
    )
    f0_file: str = Field(
        default=None
    )

    @property
    def type(self) -> str:
        """Type of the Message, used for serialization."""
        return "RVC"


@registry.register_processor("rvc_speakers")
class RVCSpeakers(BaseProcessor):
    """
    音频处理器有抽象处理器Processor，通过单独的Processor配置，预加载音频处理器，
    不同的处理器有着特定人物的说话风格与配置参数
    """

    def __init__(self, hubert_model_path: str, rvc_config_file: str):
        # Reference: https://huggingface.co/spaces/zomehwh/rvc-models/blob/main/app.py#L21  # noqa
        self.in_hf_space = getenv('SYSTEM') == 'spaces'
        self._loaded_models = []
        self._load_hubert(hubert_model_path=hubert_model_path)
        self._load_rvc_mode(rvc_config_file=rvc_config_file)

    def __call__(
            self,
            data: RvcProcessorData
    ):
        # 将 Python 列表转换为 NumPy 数组
        audio_samples_np = np.array(data.audio_samples, dtype=np.float32)
        input_audio = (data.sample_rate, audio_samples_np)

        return self.vc_func(input_audio=input_audio,
                            model_index=data.model_index,
                            f0_up_key=data.f0_up_key,
                            f0_method=data.f0_method,
                            index_rate=data.index_rate,
                            filter_radius=data.filter_radius,
                            rms_mix_rate=data.rms_mix_rate,
                            resample_sr=data.resample_sr,
                            protect=data.protect,
                            f0_file=data.f0_file)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            raise RuntimeError("from_config cfg is None.")

        hubert_model_path = cfg.get("hubert_model_path", "")
        rvc_config_file = cfg.get("rvc_config_file", "")

        return cls(hubert_model_path=os.path.join(registry.get_path("rvc_library_root"),
                                                  hubert_model_path),
                   rvc_config_file=os.path.join(registry.get_path("rvc_library_root"),
                                                rvc_config_file))

    def match(self, data: ProcessorData):
        return "RVC" in data.type

    @property
    def loaded_models(self):
        return self._loaded_models

    def _load_hubert(self, hubert_model_path: str):

        # Load hubert model
        logger.info(f'Load hubert model{hubert_model_path}')
        self.hubert_model = util.load_hubert_model(registry.get("device"), model_path=hubert_model_path)
        self.hubert_model.eval()
        logger.info('Loaded hubert model')

    def _load_rvc_mode(self, rvc_config_file: str):
        """
        模型配置加载
        :param rvc_config_file:
        :return:
        """

        # Load models
        logger.info(f'Models Load:rvc_speakers')
        multi_cfg = OmegaConf.load(get_abs_path(rvc_config_file))
        rmvpe_path = os.path.join(registry.get_path("rvc_library_root"), multi_cfg.get("rmvpe_path"))
        logger.info(f'rmvpe_path:{rmvpe_path}')
        for item in multi_cfg.get('models'):
            for key, model_info in item.items():  # 使用 .items() 方法获取键值对

                logger.info(f'Loading model: {key}')
                model_name = model_info.get("model_name")
                # Load model info
                model_info_config_file = os.path.join(registry.get_path("rvc_library_root"),
                                                      model_info.get("path"),
                                                      'config.json')

                logger.info(f'Loading model model_info_config_file: {model_info_config_file}')
                model_info_config = json.load(open(model_info_config_file, 'r'))
                # 索引文件路径
                if model_info_config.get('feat_index') is not None and model_info_config.get('feat_index') != "":
                    model_info_config['feat_index'] = os.path.join(registry.get_path("rvc_library_root"),
                                                                   model_info.get("path"),
                                                                   model_info_config['feat_index'])
                # Load RVC checkpoint
                torch_file = os.path.join(registry.get_path("rvc_library_root"),
                                          model_info.get("path"),
                                          model_info_config['model'])
                cpt = torch.load(
                    torch_file,
                    map_location='cpu'
                )
                tgt_sr = cpt['config'][-1]
                cpt['config'][-3] = cpt['weight']['emb_g.weight'].shape[0]  # n_spk

                if_f0 = cpt.get('f0', 1)
                version = cpt.get("version", "v1")

                logger.info(f'Loading model: {key},if_f0:{if_f0},version:{version}')
                synthesizer_class = {
                    ("v1", 1): SynthesizerTrnMs256NSFsid,
                    ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
                    ("v2", 1): SynthesizerTrnMs768NSFsid,
                    ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
                }

                net_g = synthesizer_class.get(
                    (version, if_f0), SynthesizerTrnMs256NSFsid
                )(*cpt["config"], is_half=util.is_half(registry.get("device")))

                del net_g.enc_q

                # According to original code, this thing seems necessary.
                logger.info(net_g.load_state_dict(cpt['weight'], strict=False))

                net_g.eval().to(registry.get("device"))
                net_g = net_g.half() if util.is_half(registry.get("device")) else net_g.float()

                vc = VC(tgt_sr,
                        registry.get("x_pad"),
                        registry.get("x_query"),
                        registry.get("x_center"),
                        registry.get("x_max"),
                        registry.get("is_half"),
                        registry.get("device"),
                        rmvpe_path=rmvpe_path
                        )

                self._loaded_models.append(dict(
                    name=model_name,
                    metadata=model_info_config,
                    vc=vc,
                    net_g=net_g,
                    if_f0=if_f0,
                    target_sr=tgt_sr,
                    version=version
                ))

        logger.info(f'Models loaded:rvc_speakers, len:{len(self._loaded_models)}')

    def vc_func(
            self,
            input_audio: Tuple[int, np.ndarray], model_index, f0_up_key, f0_method: str, index_rate,
            filter_radius, rms_mix_rate, resample_sr, protect: float = 0.33, f0_file: str = None
    ) -> Tuple[int, np.ndarray]:
        """
            # https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/blob/main/infer-web.py#L118  # noqa
        :param f0_up_key:  变调(整数, 半音数量, 升八度12降八度-12)
        :param input_audio:
        :param f0_file:  F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调
        :param protect: 保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果
        :param model_index:
        :param f0_method:
        :param index_rate: 检索特征占比
        :param filter_radius: >=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音
        :param rms_mix_rate: 输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络
        :param resample_sr: 后处理重采样至最终采样率，0为不进行重采样
        :return:
        """
        if input_audio is None:
            raise RuntimeError("Please provide input audio.")

        if model_index is None:
            raise RuntimeError("Please select a model.")

        model = self._loaded_models[model_index]

        # Reference: so-vits
        (audio_samp, audio_npy) = input_audio

        # https://huggingface.co/spaces/zomehwh/rvc-models/blob/main/app.py#L49
        # Can be change well, we will see
        if (audio_npy.shape[0] / audio_samp) > 600 and self.in_hf_space:
            raise RuntimeError("Input audio is longer than 600 secs.")

        # Bloody hell: https://stackoverflow.com/questions/26921836/
        if audio_npy.dtype != np.float32:  # :thonk:
            audio_npy = (
                    audio_npy / np.iinfo(audio_npy.dtype).max
            ).astype(np.float32)

        if len(audio_npy.shape) > 1:
            audio_npy = librosa.to_mono(audio_npy.transpose(1, 0))

        if audio_samp != 16000:
            audio_npy = librosa.resample(
                audio_npy,
                orig_sr=audio_samp,
                target_sr=16000
            )

        f0_up_key = int(f0_up_key)
        times = [0, 0, 0]

        checksum = hashlib.sha512()
        checksum.update(audio_npy.tobytes())
        feat_file_index = ''
        if (
                model['metadata']['feat_index'] != ""
                # and file_big_npy != ""
                # and os.path.exists(file_big_npy) == True
                and os.path.exists(model['metadata']['feat_index']) == True
                and index_rate != 0
        ):
            feat_file_index = model['metadata']['feat_index']

        output_audio = model['vc'].pipeline(
            self.hubert_model,
            model['net_g'],
            model['metadata'].get('speaker_id', 0),
            audio_npy,
            checksum.hexdigest(),
            times,
            f0_up_key,
            f0_method,
            feat_file_index,
            index_rate,
            model['if_f0'],
            filter_radius,
            model['target_sr'],
            resample_sr,
            rms_mix_rate,
            model['version'],
            protect,
            f0_file=f0_file
        )

        out_sr = (
            resample_sr if 16000 <= resample_sr != model['target_sr']
            else model['target_sr']
        )

        logger.info(f'npy: {times[0]}s, f0: {times[1]}s, infer: {times[2]}s')
        return out_sr, output_audio
