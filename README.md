# gpt-j-simple
 
### Project Description and Goals
This is a wrapper for kingoflolz' [mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax) for easy initialization. The vision of this project is to provide an instructive tool to welcome users that are new to the concept of sharded transformers. My goal is to provide a simple way of creating, finetuning and training meshed transformers with intuitive/descriptive documentation and source code. gpt-j-simple only supports google colab; see TODO for pemdas.
 
### List of working features
1. `download_gptj('slim')` download either gpt-j-6b full or slim
2. `sess=start_jax_sess(server='colab')` to easily initialize a sharded transformer network. Currently only accepts server='colab'. I want to adjust this to automatically detect the setup of environment, or to easily describe it.
3. `load_gptj(sess)` to load in a pretrained tranformer (GPT-j) into a sharded transformer network.
4. `generate(sess,prefix)` to generate responses.

### Installation:
install in colab with
```
!git clone https://github.com/nickthorpie/gpt-j-simple.git
!pip install -r gpt-j-simple/requirements.txt
!pip install gpt-j-simple/
```
get started with
```
from gpt_j_simple import download_gptj,start_jax_sess,load_gptj, generate
```
and follow the code in **List of working features**
### Currently working on
Right now my challenge is in writing the code for **train.py**. The kingoflolz [train.py](https://github.com/kingoflolz/mesh-transformer-jax/blob/8d26cd8cf9cc7e64a7ac18ecbe4a382ffd399691/train.py "mesh-transformer-jax/train.py") pulls from a lot of constructs that I'm unfamilliar with. Specifically, the structure of their data is a mystery, and I haven't had the time to dive into their TFRecords wrapper.

### Todo - order of operations
1. Finish up the training/finetuning sequence, which probably involves making a function to format training data for the user.
2. Once we have a working code for colab, modify our start_jax_sess to automatically detect the type of environment, find gpu's, etc. Should be all we need to support external environments.
3. Make a simple one-liner for resharding a transformer
4. implement an option for training with buckets. To the user it should be as simple setting model_dir to the `gs://bucket_name`.
5. Make documentation exceptionally clear and make sure workflow is intuitive. Clean UI is the primary objective.

### Future plans:
1. Make a straightforward plan/docs on how to create and train your own meshed tf from scratch

### Credits
This project borrows code from [kingsoflolz/mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax/) and builds on it for a more user friendly interface.
It was inspired by the project [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple) by minimaxir, and borrows its structure from there.
**please note that because this project is still under initial construction, some code from these repositories are not correcctly credited**
