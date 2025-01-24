FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install Java, Gradle, Git, Vim, and other dependencies
RUN apt-get update && apt-get install -y \
    default-jdk \
    default-jre \
    gradle \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Configure vi bindings for the terminal
RUN echo "set editing-mode vi" >> /etc/inputrc

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the ddprofiler directory into the container
COPY ddprofiler/ /app/ddprofiler/

# Run the build.sh script to build and install ddprofiler
WORKDIR /app/ddprofiler/
RUN chmod +x build.sh
RUN bash build.sh
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional Python packages
RUN pip install jupyter numpy scipy networkx pandas matplotlib black tqdm elasticsearch ipython bitarray fire neo4j

# Configure Black formatter
RUN echo "[tool.black]\nline-length = 120" > pyproject.toml

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Set the default command to run an interactive shell
# CMD ["ipython"]
