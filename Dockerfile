FROM python:3.8-slim-buster
USER root

COPY ./app /app

RUN apt-get update
RUN apt-get install nginx python3-pip python3-dev zip gcc musl-dev unzip nano systemd -y

RUN pip3 install --upgrade pip
RUN pip3 install -r /app/requirements.txt
RUN pip3 install uwsgi

COPY nlp /etc/nginx/sites-enabled/default
COPY entrypoint.sh /entrypoint.sh

EXPOSE 80
RUN chmod +x entrypoint.sh
CMD  ["./entrypoint.sh"]
