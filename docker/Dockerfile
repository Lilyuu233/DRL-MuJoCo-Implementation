FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
ENV TZ=UK \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y build-essential cmake qtbase5-dev qt5-qmake

ENV HOME /mnt/iusers01/sk01/j95445ms
ARG UID
RUN useradd -u $UID --create-home j95445ms

RUN python3 -m pip install gym3
RUN python3 -m pip install https://github.com/openai/procgen/archive/refs/tags/0.10.7.zip
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
# RUN python3 -m pip install torch_xla[tpu]==2.1.0 -f https://storage.googleapis.com/libtpu-releases/index.html

RUN echo 'set editing-mode vi' >> $HOME/.inputrc
RUN echo 'set keymap vi' >> $HOME/.inputrc
# singularity compatibility: https://github.com/qgis/QGIS/issues/46841
RUN strip --remove-section=.note.ABI-tag /usr/lib/x86_64-linux-gnu/libQt5Core.so.5

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/

WORKDIR $HOME
