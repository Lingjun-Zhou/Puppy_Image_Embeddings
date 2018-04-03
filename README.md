# PaddlePaddle SqueezeNet
An image compression model for dog images (rare puppers) that allows for similarity recognition and other semantic operations commonly associated with word embeddings. The model is composed of a SqueezeNet, with a few final layers spliced off and replaced with a set of dense layers trained in a Siamese architecture. In the examples provided in this repo, the dataset used is a breed classification corpus provided by [Stanford](http://vision.stanford.edu/aditya86/ImageNetDogs/).

(https://github.com/PaddlePaddle/models)

![license](https://img.shields.io/github/license/mashape/apistatus.svg)
[![Maintenance Intended](http://maintained.tech/badge.svg)](http://maintained.tech/)

## Requirements
You must have an AWS account and Docker-Machine installed on your computer. See Docker's documentation for [instructions](https://docs.docker.com/machine/install-machine/).
All other software will run inside an AWS EC2 instance, so you won't need any other software locally.

## Installation
These installation instructions assume you already have Docker-Machine installed on your computer.

First, provision a P2X EC2 on AWS using Docker Machine.
* Substitute your AWS credentials into `provision.sh`
* Ensure you configure a security group allowing for inbound traffic on port 8888 and update `provision.sh` accordingly. 
* `cd` into your project root (where this README is located)
* Run `bash provision.sh`

Then, you need to download the Corpus onto the new server. This line instructs the EC2 to fetch the dataset from Stanford's site.
`docker-machine ssh aws01 -- "sudo bash /home/ubuntu/utils/download_corpus.sh"`

Bring the remote ipynb online with:
Do so through docker-machine as follows (I suggest doing this inside GNU screen and swapping to a new window after running these two lines):
```
docker-machine ssh aws01 -- "sudo nvidia-docker build -t notebookserver -f Dockerfile ."
docker-machine ssh aws01 -- "sudo nvidia-docker run --name=book -v=/home/ubuntu:/book/working -p 8888:8888 notebookserver"
```
Once the notebook is live, find the public ip of your machine (use `docker-machine ip aws01`) and visit `{ip}:8888` in a browser. You should have access to the ipynb now.
The final step is to build an index for the corpus:
```
docker-machine ssh aws01 -- "sudo nvidia-docker exec -t book python /book/working/src/generate_lists.py"
```

## Train
There are two models that need to be trained: the SqueezeNet model and the Siamese model. For these next steps, you'll want to ssh into the actual ec2 using `docker-machine ssh aws01`.

### Training SqueezeNet
First, train the SqueezeNet architecture on the supervised dog breed category dataset.
If you want to resume training from an existing parameters file, comment out line 27 of the `src/category_stage/train.py` file and update lines 28 and 29 accordingly. Otherwise, comment out line 28 and 29 and uncomment line 27.
Train the model like this:
```
sudo nvidia-docker exec -t book python /book/working/src/category_stage/train.py
```
Model saves will be found in `/book/working/models/`.

In order to continue to train the Siamese model, we will need to generate some intermediary embeddings. To do this, enter `src/category_stage/generate_embeddings.py` and update line 22 with your newest model save file.
Then, run:
```
sudo nvidia-docker exec -t book python /book/working/src/category_stage/generate_embeddings.py
```
This will generate a 'dataset' of sorts for the Siamese architecture to train on. This 'dataset' will be temporarily located in `/book/working/data/` as a Numpy save file.

### Siamese model
Once the SqueezeNet has been trained, you can train the Siamese model.
Train the model like this:
```
sudo nvidia-docker exec -t book python /book/working/src/siamese_stage/train.py
```
Nothing fancy needs to be done after this step.

## Usage
To access the final result, enter the notebook in `/book/working/src/main.ipynb`.
There should be examples of embedding manipulations there, including similarity measurement and puppy algebra.

## License

Copyright 2018 Eric Zhao

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

