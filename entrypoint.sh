#!/bin/bash
service nginx start
cd app
uwsgi --ini /app/uwsgi.ini --master --uid www-data --gid www-data --lazy-apps
