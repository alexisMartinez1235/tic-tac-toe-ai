#!/bin/bash
# ---------------Parameters------------------
# $1 : force delete python installation folder (optional)

# DOCKER_BUILDKIT=1
source ./.env 

init(){
  # ---------------Parameters------------------
  # $1 : force delete python installation folder 

  local ReCreateInstallation="$1"

  # docker-compose -f "docker-compose.yml" -f "docker-compose-dev.yml" stop
  # docker-compose -f "docker-com pose.yml" stop
  
  if [[ "$ReCreateInstallation" == "true" ]]; then
    rm -rf ./Python/Installation
  fi

  mkdir -p ./VsCodeConfigFolders/Python
  
  # docker-compose -f "docker-compose.yml" -f "docker-compose-dev.yml" up -d --build
  docker-compose -f "docker-compose.yml" up -d --build
}

if [[ "true" ]]; then
  init $1
fi
