FROM python:3.10

RUN apt update && apt install -y cmake gcc portaudio19-dev ffmpeg



WORKDIR /code

ENV NUMBA_CACHE_DIR=/tmp/

ENV TRANSFORMERS_CACHE=/tmp/
ENV XDG_CACHE_HOME=/tmp/

RUN pip install --no-cache-dir poetry

COPY ./src /code/src


RUN rm -rf /code/src/poetry.lock
RUN rm -rf /code/src/projects/speakers/poetry.lock

RUN cd /code/src && \
    poetry self add poetry-multiproject-plugin

RUN cd /code/src && \
    poetry self add poetry-polylith-plugin


# vits依赖项
RUN pip install --upgrade cython numpy && \
    cd /code/src/components/speakers/vits/monotonic_align && \
    mkdir -p /code/src/components/speakers/vits/monotonic_align/vits/monotonic_align/ && \
    python setup.py build_ext --inplace && \
    mv /code/src/components/speakers/vits/monotonic_align/vits/monotonic_align/* /code/src/components/speakers/vits/monotonic_align/

# 打包服务发布包
RUN cd /code/src && \
    poetry build-project --directory projects/speakers

RUN cd /code/src/projects/speakers/dist && \
    pip install speakers-0.1.0-py3-none-any.whl

CMD ["python", "-m", "speakers.start.start", "--speakers-config-file","/media/checkpoint/RVC-Speakers-hub/speakers.yaml", "--verbose",  "--mode", "web"]

EXPOSE 10001
