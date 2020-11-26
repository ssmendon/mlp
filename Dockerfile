FROM python:3

WORKDIR /usr/src/nn

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get -y update && apt-get install -y \
    git