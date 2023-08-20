FROM python:3.10

RUN apt update && apt install -y cmake gcc portaudio19-dev

WORKDIR /code
ENV NUMBA_CACHE_DIR=/tmp/
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . /code/

RUN pip install -e .

RUN cd /code/vits/monotonic_align && \
    mkdir -p /code/vits/monotonic_align/vits/monotonic_align/ && \
    python setup.py build_ext --inplace && \
    mv /code/vits/monotonic_align/vits/monotonic_align/* /code/vits/monotonic_align/

CMD ["python", "-m", "speakers", "--verbose", "--mode", "web"]

EXPOSE 10001