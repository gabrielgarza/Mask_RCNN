#!/bin/bash

# Script taken from:
# https://docs.aws.amazon.com/batch/latest/userguide/batch-gpu-ami.html

# Install Docker
sudo yum update -y
sudo yum install -y docker

# Install ecs-init, start docker, and install nvidia-docker 2
sudo yum install -y ecs-init
sudo service docker start
DOCKER_VERSION=$(docker -v | awk '{ print $3 }' | cut -f1 -d"-")
DISTRIBUTION=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$DISTRIBUTION/nvidia-docker.repo | \
  sudo tee /etc/yum.repos.d/nvidia-docker.repo
PACKAGES=$(sudo yum search -y --showduplicates nvidia-docker2 nvidia-container-runtime | grep $DOCKER_VERSION | awk '{ print $1 }')
sudo yum install -y $PACKAGES
sudo pkill -SIGHUP dockerd

# Run test container to verify installation
sudo docker run --privileged --runtime=nvidia --rm nvidia/cuda nvidia-smi

# Update Docker daemon.json to user nvidia-container-runtime by default
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
EOF

sudo service docker restart


# Then run
# sudo docker run nvidia/cuda:latest nvidia-smi
# sudo docker rm $(sudo docker ps -aq)
# sudo docker rmi $(sudo docker images -q)
# sudo stop ecs
# sudo rm -rf /var/lib/ecs/data/ecs_agent_data.json
