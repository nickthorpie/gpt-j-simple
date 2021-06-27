from gpt_j_simple.src._utils import _download_file_with_progress,_init_colab_tpu_jax,default_params
import os
import numpy as np
from tqdm import tqdm
import xtarfile
import shutil
import json
import requests
import time

import jax
from jax.config import config
from jax.experimental import maps
import optax
try: import transformers
except: import transformers

from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer


def start_jax_sess(server='local', params={}):
    """
    Automatically a causal transformer based on your environment. Currently only set up for google colab.

    Parameters
    ----------
    server : str


    :param threads:
    :param server:
    :param params:
    :return: sess
    """
    if 'colab' in server:
        TPU = _init_colab_tpu_jax()
        nTPU = len(TPU)
        # The following is required to use TPU Driver as JAX's backend.

        params = default_params(params)
        params["sampler"] = nucleaus_sample
        params["optimizer"] = optax.scale(0)
        per_replica_batch = params['per_replica_batch']
        cores_per_replica = params["cores_per_replica"]

        mesh_shape = (nTPU // cores_per_replica, cores_per_replica)
        devices = np.array(TPU).reshape(mesh_shape)

        params["devices"] = TPU
        params["batch_count"] = per_replica_batch * jax.device_count() // cores_per_replica

        maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(devices, ('dp', 'mp')))

        sess = CausalTransformer(params)
        # sess.config.update({'params': params})
        return sess


def download_gptj(model_dir='models',
                  project_name='gptj_slim',
                  model_name='step_383500_slim',
                  ckpt = 383500,
                  overwrite=False):
    """Downloads the GPT-J model into the model_dir directory, and establishes a meta file
    from the-eye.eu.

    Parameters
    ----------
    model_dir : str
        parent directory of model to download
    model_name : str
        name of the GPT-J model to download.
        choose from (step_383500_slim, step_383500)
        As of Jun 17 2021 one for "slim" or "or full" but may later include other
        model sizes.
    overwrite : bool
        Passed to allow overwriting of current directory.

    Adapted from https://github.com/openai/gpt-2/blob/master/download_model.py
    Extracted from https://the-eye.eu/public/AI/GPT-J-6B/step_383500_slim.tar.zstd

    TODO: This is only structured to download from the original repo. As more models are released
        it may be better to make this more flexible.
    TODO: Need to make entire project more flexible for using with buckets.
    """
    if project_name is not None:
      model_dir = os.path.join(model_dir,project_name)

    models = dict.fromkeys(['slim', 'small', 'lite','step_383500_slim'], 'step_383500_slim')
    models.update(dict.fromkeys(['big', 'large','full','step_383500'], 'step_383500'))
    model_name = models[model_name]

    # approximate file sizes for our two known models.
    file_sizes = {'step_383500_slim':9*1024**3,
                  'step_383500':61*1024**3}

    # create the <model_dir>/<model_name> subdirectory if not present
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    elif overwrite==True:
      print(f"Overwriting {model_dir}")
    elif os.path.exists(model_dir):
      raise Exception("project already exists. Pass overwrite=True if you want to overwrite it")

    print(f"Downloading {model_name}")
    _download_file_with_progress(sub_dir=model_dir,
                                url=f"https://the-eye.eu/public/AI/GPT-J-6B/{model_name}.tar.zstd",
                                file_name = f"{model_name}.tar.zstd",file_size = file_sizes[model_name])

    print('Finished downloading. Unzipping now')
    with xtarfile.open(f'{model_dir}/{model_name}.tar.zstd', 'r') as archive:
      archive.extractall()

    os.remove(f'{model_dir}/{model_name}.tar.zstd')
    shutil.move('step_383500',f'{model_dir}/step_383500')

    print('writing meta file')
    with open(f"{model_dir}/meta.json", "w") as f:
        json.dump({
            "step": ckpt,
            "checkpoints": [],
            "aux": {}
        }, f)

def load_gptj(sess,
              model_dir='models',
              project_name='gptj_slim',
              ckpt='step_383500'):
    """"
    Loads a trained model into a jax transformer session.
    ----------
    create session with
    sess=start_jax_sess()
    load_gptj(sess)
    Parameters
    ----------
    sess : <jax_session object>
        This object is created with sess=start_jax_sess().

    ckpt : int or str
        Specify the checkpoint to load from as either str('step_No'), int(No), or str('latest') to get
        the most recent ckpt

    TODO: check if ckpt matches number of cores and automatically reshard
    """
    if project_name is not None:
      model_dir = os.path.join(model_dir,project_name)

    if isinstance(ckpt, str):
        if 'step' in ckpt.split('_')[0]:
            ckpt = int(ckpt.split('_')[1])

    if ckpt is 'latest':
        try:
            with open(f"{model_dir}/meta.json", "r") as f:
                meta = json.load(f)
                ckpt = meta['step']
        except:
            print('not fully done this function either')

    params = sess.config
    model_path = os.path.join(model_dir,f'step_{ckpt}')
    sess.state = read_ckpt(sess.state, model_path, params['devices'].shape[1])

    sess.state = sess.move_xmap(sess.state, np.zeros(params['cores_per_replica']))

    return sess


def generate(sess,
             prefix=None,
             truncate=None,
             nsamples=1,
             batch_size=1,
             gen_len=1023,
             temp=0.7,
             top_p=0,
             include_prefix=True,
             action='print'):
    params = sess.config['params']
    seq = params["seq"]
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
    tokens = tokenizer.encode(prefix)

    pad_amount = seq - len(tokens)

    total_batch = params['per_replica_batch'] * jax.device_count() // params['cores_per_replica']

    padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
    batched_tokens = np.array([padded_tokens] * total_batch)
    length = np.ones(total_batch, dtype=np.uint32) * len(tokens)

    start = time.time()
    output = sess.generate(batched_tokens, length, gen_len,
                           {"top_p": np.ones(total_batch) * top_p,
                            "temp": np.ones(total_batch) * temp
                            }
                           )

    samples = []
    decoded_tokens = output[1][0]

    for o in decoded_tokens[:, :, 0]:
        decoded = tokenizer.decode(o)
        if truncate in o:
            decoded = o.split(truncate, 1)[0]
        samples.append(f"\033[1m{prefix}\033[0m{decoded}")

    print(f"completion done in {time.time() - start:06}s")
    if action is 'print':
        print(samples[0])
    return samples[0]