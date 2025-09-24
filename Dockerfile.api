# Start with your base image
FROM python:3.10-slim

WORKDIR /app

COPY ./requirements-docker.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./models /app/models
COPY ./dog_and_cat_classifier_cnn_from_scratch /app/dog_and_cat_classifier_cnn_from_scratch
COPY ./api /app/api

CMD ["fastapi", "run", "api/main.py", "--port", "3000"]