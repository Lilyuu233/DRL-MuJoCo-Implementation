#!/usr/bin/env bash
docker run --gpus all -v `pwd`:/mnt/iusers01/sk01/j95445ms/  \
            -it procgen:pytorch /bin/bash

