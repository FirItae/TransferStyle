FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime
WORKDIR /app
COPY ./app /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
CMD [ "/bin/bash" ]
