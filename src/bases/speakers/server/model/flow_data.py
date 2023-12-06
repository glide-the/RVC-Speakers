from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union


class EdgeProcessorData(BaseModel):
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


class BarkProcessorData(BaseModel):
    """
        :param text: 生成文本
        :param speaker_history_prompt: 音频预设npz文件
        :param text_temp: 提示特殊标记程序，趋近于1，提示词特殊标记越明显
        :param waveform_temp: 提示隐藏空间转音频参数比例

    """
    """生成文本"""
    text: str = Field(default="你好")
    """音频预设npz文件"""
    speaker_history_prompt: str = Field(default='zh_speaker_2')
    """提示特殊标记程序，趋近于1，提示词特殊标记越明显"""
    text_temp: float = Field(default=1)
    """提示隐藏空间转音频参数比例"""
    waveform_temp: float = Field(default=0.9)


class VitsProcessorData(BaseModel):
    """生成文本"""
    text: str = Field(default="你好")
    """语言- 序号  ['日本語', '简体中文', 'English', 'Mix'] """
    language: int = Field(default=1)
    """讲话人id"""
    speaker_id: int = Field(default=0)
    """ noise_scale(控制感情变化程度)"""
    noise_scale: float = Field(default=0.5)
    """length_scale(控制整体语速)"""
    speed: int = Field(default=1)
    """noise_scale_w(控制音素发音长度)"""
    noise_scale_w: float = Field(default=1)


class RvcProcessorData(BaseModel):
    model_index: int = Field(default=0)

    """ 变调(整数, 半音数量, 升八度12降八度-12)"""
    f0_up_key: int = Field(default=0)

    """ F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调"""
    f0_method: str = Field(default="rmvpe")

    """检索特征占比"""
    index_rate: float = Field(default=0.9)
    """ >=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"""
    filter_radius: int = Field(default=1)
    """输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"""
    rms_mix_rate: int = Field(default=1)
    """后处理重采样至最终采样率，0为不进行重采样"""
    resample_sr: float = Field(default=0)
    """保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"""
    protect: float = Field(
        default=0.33
    )
    f0_file: str = Field(
        default=None
    )


class BaseFlowData(BaseModel):
    """任务创建时间"""
    created_at: float = Field(default=0)
    """任务请求时间"""
    requested_at: float = Field(default=0)


class VitsVoiceFlowData(BaseModel):
    vits: VitsProcessorData
    rvc: RvcProcessorData


class RunnerParameter(BaseModel):
    task_name: str = Field(default="vits_voice_task")


class BarkVoiceFlowData(BaseModel):
    bark: BarkProcessorData
    rvc: RvcProcessorData


class EdgeVoiceFlowData(BaseModel):
    edge: EdgeProcessorData
    rvc: RvcProcessorData


class PayLoad(BaseFlowData):
    parameter: RunnerParameter
    payload: Union[Dict, EdgeVoiceFlowData, BarkVoiceFlowData, VitsVoiceFlowData]
