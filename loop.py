import asyncio
import time
import io
import json
import argparse
from argparse import Namespace

from scipy.io.wavfile import write as write_wav, read as read_wav
import requests
import logging

from load_text import play_audio


async def runnerSubmit(data):
    # 实现异步提交任务的逻辑，使用 requests 或其他方式发送 HTTP 请求
    response = requests.post("http://localhost:10001/runner/submit", json=data)
    return response.json()


async def runnerResult(params):
    # 实现异步获取任务结果的逻辑，使用 requests 或其他方式发送 HTTP 请求
    response = requests.get("http://localhost:10001/runner/result", params=params)
    return response


async def getResultAndDisplayAudio(taskId):
    try:
        # 向服务器请求结果数据
        response = await runnerResult({"task_id": taskId})

        # 假设response.data是包含音频文件内容的字节流
        audio_data = response.content

        # 使用 io.BytesIO 将字节流转换为类文件对象
        audio_file = io.BytesIO(audio_data)

        # 读取音频文件的采样率和数据
        sample_rate, audio_data = read_wav(audio_file)

        # 返回音频数据和采样率
        return sample_rate, audio_data

    except Exception as error:
        print("获取结果时出错", error)


class MyComponent:
    def __init__(self):

        self.formData = None
        # 初始化日志记录器
        self.logger = logging.getLogger("MyComponent")
        self.logger.setLevel(logging.DEBUG)  # 设置日志级别
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 创建一个文件处理程序，将日志写入文件
        file_handler = logging.FileHandler('my_component.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # 创建一个控制台处理程序，将日志打印到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # 将处理程序添加到日志记录器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    async def submitForm(self, textInput: str):
        self.logger.info("Submitting form with text: %s", textInput)
        self.formData = {
            "created_at": 0,
            "requested_at": 0,
            "parameter": {
                "task_name": "bark_voice_task"
            },
            "payload": {
                "bark": {
                    "text": "[WOMAN] "+textInput,
                    "speaker_history_prompt": "v2/zh_speaker_9",
                    "text_temp": 0.5,
                    "waveform_temp": 0.9
                },
                "rvc": {
                    "model_index": 4,
                    "f0_up_key": 9,
                    "f0_method": "rmvpe",
                    "index_rate": 0.9,
                    "filter_radius": 1,
                    "rms_mix_rate": 1,
                    "resample_sr": 0,
                    "protect": 0.33,
                    "f0_file": ""
                }
            }
        }
        while True:
            jsonData = await runnerSubmit(self.formData)
            if jsonData.get('code') == 200 and jsonData.get('data').get("finished") is True:
                sample_rate, audio_data = await getResultAndDisplayAudio(taskId=jsonData.get('data').get("task_id"))
                return sample_rate, audio_data
            await asyncio.sleep(5)  # 异步休眠5秒

    async def text_to_speech(self, dataDict):
        for item in dataDict:
            timestamp = int(time.time())
            for key, value in item.items():
                if isinstance(value, str):
                    sample_rate, audio_data = await self.submitForm(textInput=value)
                    # save audio to disk
                    write_wav(f"wav/{key}_{timestamp}_{value}.wav", sample_rate, audio_data)
                    await play_audio(audio_data,sample_rate)
                elif isinstance(value, list):
                    for i, para in enumerate(value):
                        sample_rate, audio_data = await self.submitForm(textInput=para)
                        # save audio to disk
                        write_wav(f"wav/{key}_{timestamp}_{para}_{i}.wav", sample_rate, audio_data)
                        await play_audio(audio_data,sample_rate)


async def load_json_file(args: Namespace):
    from load_text import file_name, read_text_file
    # 创建 MyComponent 实例
    my_component = MyComponent()

    # 文件夹
    json_dir = args.file
    # 获取文件夹内的所有文件
    json_texts = file_name(json_dir)
    print(json_texts)
    for text in json_texts:
        text_content = read_text_file(text)
        data = json.loads(text_content)
        await my_component.text_to_speech(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='循环读取文件夹中的json文件，解析内容播放音频',
                                     description='S')
    parser.add_argument('-f', '--file', default='demo', type=str, help="json 文件夹")

    args = parser.parse_args()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(load_json_file(args))
