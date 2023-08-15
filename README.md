## RVC-Speakers
### 介绍

Ready Voice Controller系统采用了端到端的多引擎合成器，配备多种预处理器processors和音频处理流程audio pipeline，
系统通过微调音素、语调、音色、情感等要素，以确保实现高品质的语音生成效果。


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
