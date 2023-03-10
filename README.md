# CornerAPI

<img src="https://user-images.githubusercontent.com/33789802/224291501-cdf3ecb9-68cb-4acc-8022-6a842df391a5.jpg"  width="60%" height="60%">

API for a machine learning model trained to detect folded or torn corners and edges from scanned document images. 
The user sends the API an input image (in .jpg, .png or .tiff format) of a scanned document, and the API returns a reply 
containing the predicted classification ('ok' or 'folded_corner'), and the corresponding prediction confidence (a number
between 0 and 1).   

## Model training and testing 

The neural network model used for the image classification task was built with the Pytorch library, and the model training
was done by fine-tuning an existing [Densenet neural network model](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html).
The trained model file was transformed into the [ONNX](https://onnx.ai/) format in order to speed up inference and to make the use of the model less dependent on specific frameworks and libraries. 

The model has been trained using approximately 35 000 scanned document images, out of which 5000 images contained folded or 
torn corners and edges. With a test set of 10 000 images (1000 containing folded or torn corners and edges), the model 
reaches a 97% detection accuracy for faulty images and a 98% detection accuracy for images not containing folded or torn 
corners and edges. Both the training and test data consist mainly of document images from the 1970s onwards, digitized by the 
Finnish National Archives.

## Running the API

The API code has been built using the [FastAPI](https://fastapi.tiangolo.com/). It can be run either in a virtual environment,
or in a Docker container. Instructions for both options are given below. 

The API uses the pretrained machine learning model file located in the `/model` folder. By default the file name should be `corner_model.onnx`.
If you use a model with different name, you need to update the model name in the `MODEL_PATH` variable of the `api.py` file.

### Running the API in a virtual environment

These instructions use a conda virtual environment, and as a precondition you should have Miniconda or Anaconda installed on your operating system. 
More information on the installation can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). 

First create and activate conda environment using the following commands:

`conda create -n corner_api_env python=3.7`

`conda activate corner_api_env`

Next install dependencies listed in the requirements.txt file:

`pip install -r requirements.txt`

You can start the API running a single process (with Uvicorn server):

- Using default host: 0.0.0.0, default port: 8000

`uvicorn api:app`

- Select different host / port:

`uvicorn api:app --host 0.0.0.0 --port 8080`

You can also start the API with Gunicorn as the process manager (find more information [here](https://fastapi.tiangolo.com/deployment/server-workers/)):

`gunicorn api:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080`

- workers: The number of worker processes to use, each will run a Uvicorn worker

- worker-class: The Gunicorn-compatible worker class to use in the worker processes

- bind: This tells Gunicorn the IP and the port to listen to, using a colon (:) to separate the IP and the port

### Running the API using Docker

As a precondition, you should have Docker Engine installed. More information on the installation can be found [here](https://docs.docker.com/engine/install/). 

Build Docker image using the Dockerfile included in the repository: 

`docker build -t corner_image .`

Here the new image is named corner_image. After successfully creating the image, you can find it in the list of images by typing `docker image ls`.

Create and run a container based on the image:

`sudo docker run -d --name corner_container -p 8000:8000 corner_image`

In the Dockerfile, port 8000 is exposed, meaning that the container listens to that port. 
In the above command, the corresponding host port can be chosen as the first element in `-p <host-port>:<container-port>`. 
If only the container port is specified, Docker will automatically select a free port as the host port. The port mapping of the container can be viewed with the command

`sudo docker port corner_container`

## Testing the API

The API has two endpoints: `/corner` endpoint expects the input image to be included in the client's POST request, while  
`/cornerurl` endpoint expects to receive the  filepath to the image as a query parameter.

### Testing the API in a virtual environment

You can test the `/corner` endpoint of the API for example using curl:

`curl http://127.0.0.1:8000/corner -F file=@/path/img.jpg`

The second option is to send the url/path to the image file with the http request:

`curl http://127.0.0.1:8000/cornerurl?url=/path/img.jpg`

The host and port should be the same ones that were defined when starting the API.
The image path `/path/img.jpg` should be replaced with a path to the image that is used as test input. 

### Testing the API using Docker

In the Docker version of the API, the use of the latter option for passing input to the API requires 
you to use [bind mount](https://docs.docker.com/storage/bind-mounts/) to mount the desired file or 
directory into the Docker container. For instance if your input images are located in a local folder 
`/home/user/data` and you want to pass their filepaths to the containerized API, you can create and start the 
container with the command 

`docker run -v /home/user/data:/data -d --name corner_container -p 8000:8000 corner_image`

and then send the image paths to the API with the http request:

`curl http://127.0.0.1:8000/cornerurl?url=/data/img.jpg`

### Output of the API

The output is in a .json form and consists of the predicted class label and the confidence for the prediction.
So for instance the output could be 

`{"prediction":"folded_corner","confidence":0.995205283164978}`


