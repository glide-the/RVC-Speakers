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
            input_audio, model_index, pitch_adjust, f0_method, feat_ratio,
            filter_radius, rms_mix_rate, resample_option
    ):
        """
            # https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/blob/main/infer-web.py#L118  # noqa
        :param model_index:
        :param pitch_adjust:
        :param f0_method:
        :param feat_ratio:
        :param filter_radius:
        :param rms_mix_rate:
        :param resample_option:
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

        pitch_int = int(pitch_adjust)

        resample = (
            0 if resample_option == 'Disable resampling'
            else int(resample_option)
        )

        times = [0, 0, 0]

        checksum = hashlib.sha512()
        checksum.update(audio_npy.tobytes())

        output_audio = model['vc'].pipeline(
            self.hubert_model,
            model['net_g'],
            model['metadata'].get('speaker_id', 0),
            audio_npy,
            checksum.hexdigest(),
            times,
            pitch_int,
            f0_method,
            path.join('model', model['name'], model['metadata']['feat_index']),
            feat_ratio,
            model['if_f0'],
            filter_radius,
            model['target_sr'],
            resample,
            rms_mix_rate,
            'v2'
        )

        out_sr = (
            resample if resample >= 16000 and model['target_sr'] != resample
            else model['target_sr']
        )

        print(f'npy: {times[0]}s, f0: {times[1]}s, infer: {times[2]}s')
        return out_sr, output_audio
