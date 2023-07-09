#!/usr/bin/env bash

IMAGE_NAME="transfer_style:bot"

SCRIPTS=$( realpath $( dirname ${BASH_SOURCE[0]} ) )
PROJECT_DIR=$(dirname $SCRIPTS)
DOCKERFILE_PATH="${PROJECT_DIR}/Dockerfile"

docker build -f ${DOCKERFILE_PATH} -t ${IMAGE_NAME} ${PROJECT_DIR}
