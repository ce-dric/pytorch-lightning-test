
## Create Dockerfile

```dockerfile
# https://hub.docker.com/r/pytorch/pytorch/tags
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel
COPY requirements/txt /workspace/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
```

## Build docker image

```shell
> cd path/to/Dockefile
> docker build -t torch2 .
```

## Execute container

| Env | Path |
| -- | -- |
| command prompt | %CD% |
| powershell | ${pwd} |
| ubuntu | $(pwd) |

```shell
> docker run -it --name [NAME] -p 8888:8888 --gpus all --shm-size=8G -v %PATH%:/workspace torch2
# example - ubuntu
# docker run -it --name [NAME] -p 8888:8888 --gpus all --shm-size=8G -v $(pwd):/workspace torch2
```

## Notebook

```shell
jupyter notebook --ip 0.0.0.0 --allow-root --no-browser
```