#!/usr/bin/env bash

IMAGE_NAME="transfer_style:bot"
CONTAINER_NAME="transfer_style_dev"

SCRIPTS=$( realpath $( dirname ${BASH_SOURCE[0]} ) )
PROJECT_DIR=$(dirname $SCRIPTS)

echo "Recreating container..."
docker stop ${CONTAINER_NAME} && docker rm ${CONTAINER_NAME}
echo "Running new container"
docker run -it -d \
           --runtime=nvidia \
           -v ${PROJECT_DIR}/app:/app \
           --name ${CONTAINER_NAME} ${IMAGE_NAME}