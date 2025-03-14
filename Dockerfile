FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install Java, Gradle, Git, Vim, and other dependencies
RUN apt-get update && apt-get install -y \
    default-jdk \
    default-jre \
    curl \
    gradle \
    git \
    vim \
    wget \
    gnupg \
    && wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | gpg --dearmor > /usr/share/keyrings/elasticsearch-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/elasticsearch-keyring.gpg] https://artifacts.elastic.co/packages/7.x/apt stable main" | tee /etc/apt/sources.list.d/elastic-7.x.list \
    && apt-get update && apt-get install -y elasticsearch \
    && rm -rf /var/lib/apt/lists/*


# Change the terminal prompt color
RUN echo 'PS1="\[\e[36m\]\u@\h:\w\$\[\e[m\] "' >> /root/.bashrc

# Configure vi bindings for the terminal
RUN echo "set editing-mode vi" >> /etc/inputrc

# Configure Vim history and undofile
RUN echo "set history=1000" >> /etc/vim/vimrc \
    && echo "set undofile" >> /etc/vim/vimrc


# Configure Elasticsearch
RUN echo "network.host: 0.0.0.0" >> /etc/elasticsearch/elasticsearch.yml \
    && echo "discovery.type: single-node" >> /etc/elasticsearch/elasticsearch.yml \
    && echo "http.port: 9200" >> /etc/elasticsearch/elasticsearch.yml \
    && echo "transport.port: 9300" >> /etc/elasticsearch/elasticsearch.yml


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

# Initialize pre-commit formatting
RUN pre-commit install

# Make port available to the world outside this container
EXPOSE 9200
EXPOSE 9300

# Define environment variable
# ENV NAME World

# Set the default command to run an interactive shell
# CMD ["ipython"]

# Start Elasticsearch manually as a background service
# CMD service elasticsearch start && tail -f /dev/null
# Replace your current CMD with this
CMD ["/bin/bash", "-c", "service elasticsearch start && bash"]
