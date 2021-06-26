# gpt-j-simple
 
This is a wrapper for kingoflolz' [mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax) for easy initialization. The goal of this project is to provide a simple way of creating, finetuning and training meshed transformers.
 
### List of working features
1. download either gpt-j-6b full or slim with `download_gptj('slim')`
2. Easily initialize a sharded transformer network with `sess=start_jax_sess(server='colab')`. Currently only accepts server='colab'. I want to adjust this to automatically detect the setup of environment, or to easily describe it.
3. Load in a pretrained tranformer (GPT-j) into a sharded transformer network using `load_gptj(sess)`
4. Generate responses with `generate(sess,prefix)`.

### Currently working on
Right now my challenge is in writing the code for **train.py**. The kingoflolz [train.py](https://github.com/kingoflolz/mesh-transformer-jax/blob/8d26cd8cf9cc7e64a7ac18ecbe4a382ffd399691/train.py mesh-transformer-jax/train.py) pulls from a lot of constructs that I'm unfamilliar with. Specifically, the structure of their data is a mystery, and I haven't had the time to dive into their TFRecords wrapper.

### Todo
1. Finish up the training/finetuning sequence, which probably involves making a function to format training data
2. Once we have a working code for colab, modify our start_jax_sess to automatically detect the type of environment, find gpu's, etc.
3. Make a simple one-liner for resharding a transformer
4. implement an option for training with buckets. To the user it should be as simple setting model_dir to the `gs://bucket_name`.

### Credits
This project borrows code from [kingsoflolz/mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax/) and builds on it for a more user friendly interface.
It was inspired by the project [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple) by minimaxir, and borrows its structure from there.
**please note that because this project is still under initial construction, some code from these repositories are not correcctly credited**
