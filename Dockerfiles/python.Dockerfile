ARG python_version
FROM python:${python_version} as dev
  WORKDIR /usr/src/app
  
  ## Create user ##
  RUN adduser -D worker
  #################
  
  RUN pip install --upgrade pip
  RUN pip install pipenv
  
  ADD --chown=worker:worker ./Python/w/Pipfile .
  ADD --chown=worker:worker ./Python/w/Pipfile.lock .
  
  RUN pipenv install
  
  CMD python init.py