import os

import os
import pyaudio
import numpy as np
import asyncio




def file_name(file_dir):
    """
    获取输入文件夹内的所有wav文件，并返回文件名全称列表
    """
    L = []
    for root, dirs, files in os.walk(file_dir):
        print(files)
        for file in files:
            if os.path.splitext(file)[1] == '.json':
                filename = os.path.join(root, file)
                L.append(filename)
    return L


def read_text_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None
    except Exception as e:
        print("An error occurred while reading the file:", e)
        return None


# 出错Could not import the PyAudio C module 'pyaudio._portaudio'.
# 运行下面命令
# conda install -c anaconda portaudio

async def play_audio(audio_data, rate):
    p = pyaudio.PyAudio()

    def callback(in_data, frame_count, time_info, status):
        nonlocal audio_data
        data = audio_data[:frame_count]
        audio_data = audio_data[frame_count:]
        return (data.tobytes(), pyaudio.paContinue)

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    output=True,
                    stream_callback=callback)

    stream.start_stream()
    while stream.is_active():
        await asyncio.sleep(0.3)

    stream.stop_stream()
    stream.close()
    p.terminate()
