NVIDIA_SMI_TEST := $(shell nvidia-smi -L >> gpus.txt; wc -l < gpus.txt; rm gpus.txt)
ifneq ($(NVIDIA_SMI_TEST), 0)
GPUS=--gpus all
else
GPUS=
endif

# Set flag for docker run command
BASE_FLAGS=-itd --rm  -v ${PWD}:/home/workdir --shm-size 20G --name ${USER}.mlmc4rltmp
RUN_FLAGS=$(GPUS) $(BASE_FLAGS)

DOCKER_IMAGE_NAME = mlmc4rl
IMAGE = $(DOCKER_IMAGE_NAME):latest
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)
USE_CUDA = $(if $(GPUS),true,false)

# make file commands
build:
	DOCKER_BUILDKIT=1 docker build --build-arg USE_CUDA=$(USE_CUDA) --tag $(IMAGE) --progress=plain ${PWD}/.

run:
	# $(DOCKER_RUN) /bin/bash 
	docker run -itd --gpus all --shm-size=32768m --name ${USER}-container-mlmc \
		   --restart=always -v ${PWD}:/root -p 8183:8888 \
		   ${USER}-base-image 

