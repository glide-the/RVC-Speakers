## RVC-Speakers
### 介绍

我们项目采用多种TTS技术，将文本转为自然语音。
通过rvc模型，我们能合成特定人物声音。项目专注于创造沉浸式声音体验
我们通过调节音素、语调、音色、情感等要素， 确保高质量语音生成。


### 运行
#### 环境
- python 3.10
```shell
pip install -r requirements.txt
```

#### VITS
- 代码来源 https://github.com/Plachtaa/VITS-fast-fine-tuning.git

- 前置依赖
```shell
$ cd vits/monotonic_align
$ mkdir -p vits/monotonic_align/
$ python setup.py build_ext --inplace
$ mv vits/monotonic_align/* .

```
- 模块
``` 
vits目录
```
 
#### RVC

- 代码来源  https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI

- 模块
``` 
rvc目录
```
