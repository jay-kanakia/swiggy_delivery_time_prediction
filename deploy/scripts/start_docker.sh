#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 901619351636.dkr.ecr.us-east-1.amazonaws.com

echo "Pulling Docker image..."
docker pull 901619351636.dkr.ecr.us-east-1.amazonaws.com/swiggy-ecr:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=swiggy_time_pred)" ]; then
    echo "Stopping existing container..."
    docker stop swiggy_time_pred
fi

if [ "$(docker ps -aq -f name=swiggy_time_pred)" ]; then
    echo "Removing existing container..."
    docker rm swiggy_time_pred
fi

echo "Starting new container..."
docker run -d -p 80:8000 --name swiggy_time_pred -e DAGSHUB_TOKEN=db9afac195aa7f97c0ce2afc822374fa7d87fd39 901619351636.dkr.ecr.us-east-1.amazonaws.com/swiggy-ecr:latest

echo "Container started successfully."