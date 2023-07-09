#!/usr/bin/env bash

IMAGE_NAME="transfer_style:bot"
CONTAINER_NAME="transfer_style_bot"

SCRIPTS=$( realpath $( dirname ${BASH_SOURCE[0]} ) )
PROJECT_DIR=$(dirname $SCRIPTS)

echo "Recreating container..."
docker stop ${CONTAINER_NAME} && docker rm ${CONTAINER_NAME}
echo "Running new container"
docker run -d --name ${CONTAINER_NAME} -e BOT_TOKEN=${BOT_TOKEN} ${IMAGE_NAME} python app.py
