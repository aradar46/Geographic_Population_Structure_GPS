# Use a base image with Conda pre-installed
FROM continuumio/miniconda3:latest

# Copy all files from the host to the container
COPY . /app

# Create and activate the Conda environment
RUN conda create --name GPS-env -y
RUN conda env update --name GPS-env --file /app/conda_environment.yml
RUN echo "conda activate GPS-env" >> ~/.bashrc

# Install Streamlit
RUN /bin/bash -c "source activate GPS-env"

# Expose port 8501 for Streamlit
EXPOSE 8501

# Set the working directory
WORKDIR /app

# 2.Steps
# 2.1 Install Docker
# Make sure Docker is installed, otherwise you need to google &quot;how to install docker&quot; or check this
# page (https://docs.docker.com/engine/install/centos/).

# Build the image (need to be in the same directory as the Dockerfile)
# docker build -t gps-app . (the dot is important)
 

# Run the container with port forwarding
# docker run -itd --name gps-container -p 8501:8501 gps-image (to run in detached mode)
# docker attach docked-id (to attach to the container)
 
# Run Streamlit with port forwarding
# streamlit run --server.port 8501 app.py
# broswer: http://localhost:8501/


# Run Streamlit with port forwarding and in detached mode
# streamlit run --server.port 8501 app.py &


# docker rmi $(docker images -aq)
# docker rm $(docker ps -aq)

