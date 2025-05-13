FROM continuumio/miniconda

WORKDIR /sketchy_PRJ

COPY . /sketchy_PRJ/

RUN conda config --env --add channels conda-forge
RUN conda create --name general_env_docker --file ./requirements.txt

RUN conda activate general_env_docker

EXPOSE 9999

CMD [ "python", "main.py" ]
