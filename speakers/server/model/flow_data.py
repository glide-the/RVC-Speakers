from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class VitsProcessorData(BaseModel):
    """生成文本"""
    text: str
    """语言- 0中文， 1 日文"""
    language: int
    """讲话人id"""
    speaker_id: int
    """ noise_scale(控制感情变化程度)"""
    noise_scale: float
    """length_scale(控制整体语速)"""
    speed: int
    """noise_scale_w(控制音素发音长度)"""
    noise_scale_w: float


class RvcProcessorData(BaseModel):
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
    rms_mix_rate: int
    """后处理重采样至最终采样率，0为不进行重采样"""
    resample_sr: float
    """保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"""
    protect: float = Field(
        default=0.33
    )
    f0_file: str = Field(
        default=None
    )


class BaseFlowData(BaseModel):
    """任务创建时间"""
    created_at: float
    """任务请求时间"""
    requested_at: float


class VoiceFlowData(BaseFlowData):
    vits: VitsProcessorData
    rvc: RvcProcessorData
