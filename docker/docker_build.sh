#!/usr/bin/env bash
docker build --build-arg UID=$UID \
             -t procgen:pytorch . 

# docker login
# docker tag procgen:pytorch mingfeisun/procgen:pytorch
# docker push mingfeisun/procgen:pytorch
