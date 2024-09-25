# MLOps Learning Repository

## Overview

This repository is dedicated to learning and applying **MLOps** practices for streamlining machine learning workflows. It contains key files and scripts necessary for automating model development, deployment, and monitoring.

## Key Features

- **Dockerfile**: Containerization setup for ML models.
- **CI/CD pipelines**: Configuration for automating the testing and deployment of models.
- **Deployment templates**: Scripts and templates for deploying machine learning models to cloud or local environments.

## Docker commands
1. To build the docker file (move to the docker file directory , use (docker login))
```
docker build -t gemi-anpr .
```
2. Tag the image
```
docker tag gemi-anpr username/gemi-anpr:latest
```
3. Push the image into
```
docker push username/gemi-anpr:latest
```
 
## Setup and Run Instructions
 
1. Create a directory for the ANPR system:
   ```
   mkdir ~/gemi-anpr
   ```
 
2. Pull the Docker image:
   ```
   docker pull mvramkumar/gemi-anpr:latest
   ```
 
3. Run the container:
   ```
   docker run --name gemi-anpr -d \
     -v ~/gemi-anpr:/app/data \
     mvramkumar/gemi-anpr:latest /app/data
   ```
 
4. Check the logs to see if it's running with dummy RTSP links:
   ```
   docker logs gemi-anpr
   ```
 
5. If you see a message about dummy RTSP links, edit the config file:
   ```
   nano ~/gemi-anpr/config.ini
   ```
   Update the RTSPLink1 and RTSPLink2 with your actual RTSP stream URLs.
 
6. Restart the container to apply changes:
   ```
   docker restart gemi-anpr
   ```
 
7. Check the logs again to ensure it's running with your RTSP links:
   ```
   docker logs gemi-anpr
   ```
 
Your ANPR system should now be running with your specified RTSP streams. The processed plates will be saved in `~/gemi-anpr/clean_plates`.
