def _download_file_with_progress(sub_dir,url,file_name,file_size=1024**3):
    """General utility for incrementally downloading files from the internet
    with progress bar
    from url_base / sub_dir / filename
    to local file system sub_dir / filename
    Parameters
    ----------
    file_name : str
        name of file to get e.g. "hparams.json"
    sub_dir: str
        subdirectory inside which to get and copy locally eg. "models/124M"
        no trailing slash
    url_base : str
        Start of URL location specifying server and any base directories no
        trailing slash
        e.g. "https://storage.googleapis.com/gpt-2"
    """

    # set to download 1GB at a time.
      # smaller chunk size equals lighter ram performance
      # larger chunk size equals fewer server calls
    DOWNLOAD_CHUNK_SIZE = 1024**3
    r = requests.get(url, stream=True)
    with open(os.path.join(sub_dir, file_name), 'wb') as f:
        with tqdm(ncols=100, desc="Fetching " + file_name,
                  total=file_size, unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)
                pbar.update(DOWNLOAD_CHUNK_SIZE)

def _init_colab_tpu_jax():
  """In colab, if we request the tpu several times, xla freezes"""
  try:
    devices = jax.devices('tpu')
  except:
    colab_tpu_addr = os.environ['COLAB_TPU_ADDR'].split(':')[0]
    url = f'http://{colab_tpu_addr}:8475/requestversion/tpu_driver0.1_dev20210607'
    requests.post(url)
    config.FLAGS.jax_xla_backend = "tpu_driver"
    config.FLAGS.jax_backend_target = "grpc://" + os.environ['COLAB_TPU_ADDR']
    devices = jax.devices('tpu')
  return(devices)

def get_available_gpus():
  return str(jax.get_devices('gpu'))

def get_available_tpus():
  return str(jax.get_devices('tpu'))

def default_params(params=None):
    default_params ={
      "layers": 28,
      "d_model": 4096,
      "n_heads": 16,
      "n_vocab": 50400,
      "norm": "layernorm",
      "pe": "rotary",
      "pe_rotary_dims": 64,

      "seq": 2048,
      "cores_per_replica": 8,
      "per_replica_batch": 1,
      "gradient_accumulation_steps": 16,

      "warmup_steps": 3000,
      "anneal_steps": 300000,
      "lr": 1.2e-4,
      "end_lr": 1.2e-5,
      "weight_decay": 0.1,
      "total_steps": 350000,

      "tpu_size": 256,

      "bucket": "neo-models",
      "model_dir": "mesh_jax_pile_6B_rotary",

      "train_set": "pile.train.index",
      "val_set": {
        "pile": "pile.val.index",
        "owt": "openwebtext2_new_inputs.val.index"
      },

      "eval_harness_tasks": [
        "lambada",
        "piqa",
        "hellaswag",
        "winogrande",
        "mathqa",
        "pubmedqa"
      ],

      "val_batches": 100,
      "val_every": 500,
      "ckpt_every": 500,
      "keep_every": 10000,

      "name": "GPT3_6B_pile_rotary",
      "comment": ""
      }
    if params: default_params.update(params)
    return default_params