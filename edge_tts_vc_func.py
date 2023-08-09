import edge_tts
import asyncio
import util
import numpy as np
import librosa
import hashlib
import config
import json

import torch
from infer_pack.models import (
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono
)
from os import path, getenv
from typing import Union
from vc_infer_pipeline import VC


class RVCSpeakers:

    def __init__(self, hubert_model_path: str, rvc_config_file: str):
        import nest_asyncio
        nest_asyncio.apply()
        # Edge TTS speakers
        self.tts_speakers_list = asyncio.get_event_loop().run_until_complete(edge_tts.list_voices())  # noqa
        # Reference: https://huggingface.co/spaces/zomehwh/rvc-models/blob/main/app.py#L21  # noqa
        self.in_hf_space = getenv('SYSTEM') == 'spaces'
        self._load_hubert(hubert_model_path=hubert_model_path)
        self._load_rvc_mode(rvc_config_file=rvc_config_file)

    def _load_hubert(self, hubert_model_path: str):

        # Load hubert model
        self.hubert_model = util.load_hubert_model(config.device, model_path=hubert_model_path)
        self.hubert_model.eval()

    def _load_rvc_mode(self, rvc_config_file: str):

        # Load models
        multi_cfg = json.load(open(rvc_config_file, 'r'))
        self.loaded_models = []

        for model_name in multi_cfg.get('models'):
            print(f'Loading model: {model_name}')

            # Load model info
            model_info = json.load(
                open(path.join('model', model_name, 'config.json'), 'r')
            )

            # Load RVC checkpoint
            cpt = torch.load(
                path.join('model', model_name, model_info['model']),
                map_location='cpu'
            )
            tgt_sr = cpt['config'][-1]
            cpt['config'][-3] = cpt['weight']['emb_g.weight'].shape[0]  # n_spk

            if_f0 = cpt.get('f0', 1)
            net_g: Union[SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono]
            if if_f0 == 1:
                net_g = SynthesizerTrnMs768NSFsid(
                    *cpt['config'],
                    is_half=util.is_half(config.device)
                )
            else:
                net_g = SynthesizerTrnMs768NSFsid_nono(*cpt['config'])

            del net_g.enc_q

            # According to original code, this thing seems necessary.
            print(net_g.load_state_dict(cpt['weight'], strict=False))

            net_g.eval().to(config.device)
            net_g = net_g.half() if util.is_half(config.device) else net_g.float()

            vc = VC(tgt_sr, config)

            self.loaded_models.append(dict(
                name=model_name,
                metadata=model_info,
                vc=vc,
                net_g=net_g,
                if_f0=if_f0,
                target_sr=tgt_sr
            ))

        print(f'Models loaded: {len(self.loaded_models)}')

    def vc_func(
            self,
            input_audio, model_index, f0_up_key, f0_method, index_rate,
            filter_radius, rms_mix_rate, resample_sr, protect: float = 0.33, f0_file: str = None
    ):
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
            return (None, 'Please provide input audio.')

        if model_index is None:
            return (None, 'Please select a model.')

        model = self.loaded_models[model_index]

        # Reference: so-vits
        (audio_samp, audio_npy) = input_audio

        # https://huggingface.co/spaces/zomehwh/rvc-models/blob/main/app.py#L49
        # Can be change well, we will see
        if (audio_npy.shape[0] / audio_samp) > 600 and self.in_hf_space:
            return (None, 'Input audio is longer than 600 secs.')

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
        print(f0_up_key)
        times = [0, 0, 0]

        checksum = hashlib.sha512()
        checksum.update(audio_npy.tobytes())
        feat_file_index = ''
        if (
                model['metadata']['feat_index'] != ""
                # and file_big_npy != ""
                # and os.path.exists(file_big_npy) == True
                and os.path.exists( model['metadata']['feat_index']) == True
                and index_rate != 0
        ):
            feat_file_index = path.join('model', model['name'], model['metadata']['feat_index'])

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
            'v2',
            protect,
            f0_file=f0_file
        )

        out_sr = (
            resample_sr if 16000 <= resample_sr != model['target_sr']
            else model['target_sr']
        )

        print(f'npy: {times[0]}s, f0: {times[1]}s, infer: {times[2]}s')
        return out_sr, output_audio
