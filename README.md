# MLOps Learning Repository

## Overview

This repository is dedicated to learning and applying **MLOps** practices for streamlining machine learning workflows. It contains key files and scripts necessary for automating model development, deployment, and monitoring.

## Key Features

- **Dockerfile**: Containerization setup for ML models.
- **CI/CD pipelines**: Configuration for automating the testing and deployment of models.
- **Deployment templates**: Scripts and templates for deploying machine learning models to cloud or local environments.

## Docker commands

```bash
docker pull mvramkumar/anpr:latest
mkdir -p ~/anpr/saved_plates ~/anpr/clean_plates
docker run --name anpr -d \
  -e CAMERA1='rtsp://admin:msfconsole1%24@192.168.30.25:554/onvif1' \
  -e CAMERA2='rtsp://admin:msfconsole1%24@192.168.30.27:554/onvif1' \
  -e DETECTION_INTERVAL=5 \
  -v ~/anpr/saved_plates:/app/saved_plates \
  -v ~/anpr/clean_plates:/app/clean_plates \
  mvramkumar/anpr:latest
docker logs anpr
```
