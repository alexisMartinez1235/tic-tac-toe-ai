ARG python_version
FROM python:${python_version} as dev
  WORKDIR /usr/src/app

  VOLUME ./Python /usr/src/app

  # RUN apk update
  RUN apk add --virtual gcc

  # app vscode
  RUN apk add libgcc libstdc++

  ## Create user ##
  RUN adduser -D worker
  RUN apk add --update sudo
  RUN apk add htop
  RUN apk add --update make cmake gcc g++ gfortran
  
  # RUN touch /etc/sudoers.d/worker
  RUN echo "worker ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/worker \
          && chmod 0440 /etc/sudoers.d/worker
          
  RUN chown worker:root -R /home/worker/
  #################

  ADD --chown=worker:root ./Python/requirements.txt .
  RUN chown worker:root -R /usr/src/app
  # RUN pip install virtualenv
  # RUN virtualenv -p /usr/bin/python3 venv/ ; . venv/bin/activate

  # RUN python3 -m pip install --upgrade pip
  # RUN python3 -m pip install --no-cache-dir -r requirements.txt

  RUN python -m venv /opt/venv/
  RUN chown worker:root -R /opt/venv/
  RUN /opt/venv/bin/python -m pip install --upgrade pip
  RUN /opt/venv/bin/pip install --no-cache-dir -r  requirements.txt

  # CMD . /opt/venv/bin/activate && exec /opt/venv/bin/python3 -m debugpy --listen localhost:5678 --wait-for-client main.py
  # CMD /opt/venv/bin/python -m debugpy --listen localhost:5678 --wait-for-client main.py
  CMD /opt/venv/bin/python -m debug.py main.py

# FROM python:${python_version} as prod
