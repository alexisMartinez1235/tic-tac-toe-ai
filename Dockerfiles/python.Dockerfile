ARG python_version
FROM python:${python_version} as dev
  WORKDIR /usr/src/app

  # RUN apk update
  RUN apk add --virtual gcc
  
  ## Create user ##
  RUN adduser -D worker
  RUN apk add --update sudo
  RUN apk add htop
  
  # RUN touch /etc/sudoers.d/worker
  RUN echo "worker ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/worker \
          && chmod 0440 /etc/sudoers.d/worker
          
  RUN chown worker:worker -R /home/worker/
  #################

  ADD --chown=worker:worker ./Python/src/requirements.txt .

  RUN chown worker:root -R /usr/src/app
  RUN python3 -m venv /opt/venv
  RUN /opt/venv/bin/python3 -m pip install --upgrade pip
  RUN . /opt/venv/bin/activate && pip install --no-cache-dir -r requirements.txt
  RUN chown worker:root -R /opt/venv/

  # CMD sleep 10000
  CMD . /opt/venv/bin/activate && exec /opt/venv/bin/python3 -m debugpy --listen localhost:5678 --wait-for-client main.py || sleep 10000

# FROM python:${python_version} as prod
