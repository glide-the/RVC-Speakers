## RVC-Speakers
### 介绍

Ready Voice Controller系统采用了端到端的多引擎合成器，配备多种预处理器processors和音频处理流程audio pipeline，
可支持微调音素、语调、音色、情感等要素，控制语音生成效果。

### 演示地址
https://huggingface.co/spaces/dmeck/RVC-Speakers


### 运行
#### 环境
项目基于python3.10开发，建议使用python3.10运行，构建环境使用poetry管理，需要安装poetry
- python 3.10

 
#### poetry

```shell
pip install --no-cache-dir poetry       
```

> 依赖插件

- poetry-multiproject-plugin
```text
https://pypi.org/project/poetry-multiproject-plugin/
```

- poetry-polylith-plugin
```text
https://davidvujic.github.io/python-polylith-docs/commands/
```

### 模型与配置文件

#### 配置文件
    speakers/speakers.yaml

#### 模型文件
模型存放在/media/checkpoint/RVC-Speakers-hub目录下，
如果没有该目录，需要创建该目录，然后下载模型文件到该目录下
文件结构如下
```text
/media/checkpoint >>> tree -L 3 --dirsfirst RVC-Speakers-hub                                                           

RVC-Speakers-hub
├── bark
│   ├── assets
│   │   └── prompts
│   └── model
│       ├── bert-base-multilingual-cased
│       ├── codec
│       └── suno
├── rvc
│   ├── model
│   │   ├── arianagrande
│   │   ├── bob
│   │   ├── lulu
│   │   ├── maomao
│   │   ├── syz
│   │   ├── yiqing
│   │   ├── hubert_base.pt
│   │   └── rmvpe.pt
│   └── rvc.yaml
├── vits
│   └── model
│       ├── config.json
│       ├── D_0-p.pth
│       ├── D_0.pth
│       ├── G_0-p.pth
│       ├── G_0.pth
│       └── G_953000.pth
└── speakers.yaml

18 directories, 10 files

```

#### VITS
- 代码来源 https://github.com/Plachtaa/VITS-fast-fine-tuning.git

- 模块所属vits目录
 
#### RVC

- 代码来源  https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI

- 模块所属rvc目录

- 模型配置下面两个模型需要下载到rvc/model目录下
``` 

[hubert_base.pt](rvc/model/hubert_base.pt)
下载地址：https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt

[rmvpe.pt](rvc/model/rmvpe.pt)
如果你想使用最新的RMVPE人声音高提取算法，则你需要下载音高提取模型 
下载地址：https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt
```
#### bark

- 代码来源  https://github.com/suno-ai/bark

- 模块所属bark目录

- 预设的prompt文件，需要放在下面的目录下
``` 
[prompts](bark/assets/prompts)
下载地址：https://github.com/suno-ai/bark/tree/main/bark/assets/prompts
```


### 如何运行


#### Create a virtual environment and install the dependencies:
```shell
poetry install
```


#### 启动命令
任选一种

1、模块启动
- Run the module:
```shell
python -m speakers.start.start --speakers-config-file /media/checkpoint/RVC-Speakers-hub/speakers.yaml  --verbose --mode web   
```

2、直接启动
```shell
python src/bases/speakers/start/start.py --speakers-config-file /media/checkpoint/RVC-Speakers-hub/speakers.yaml --verbose --mode web
```


### 常见问题

- vits存在前置依赖，需要执行以下命令
```shell
pip install --upgrade cython numpy && \
    cd src/components/speakers/vits/monotonic_align && \
    mkdir -p src/components/speakers/vits/monotonic_align/vits/monotonic_align/ && \
    python setup.py build_ext --inplace && \
    mv src/components/speakers/vits/monotonic_align/vits/monotonic_align/* src/components/speakers/vits/monotonic_align/RUN pip install --upgrade cython numpy && \
    cd src/components/speakers/vits/monotonic_align && \
    mkdir -p src/components/speakers/vits/monotonic_align/vits/monotonic_align/ && \
    python setup.py build_ext --inplace && \
    mv src/components/speakers/vits/monotonic_align/vits/monotonic_align/* src/components/speakers/vits/monotonic_align/

```
