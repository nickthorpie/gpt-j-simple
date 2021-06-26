from gpt_j_simple.src._utils import _download_file_with_progress,_init_colab_tpu_jax,default_params

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
    download_file_with_progress(sub_dir=model_dir,
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

    models = dict.fromkeys(['slim', 'small', 'lite', 'step_383500_slim'], 'step_383500_slim')
    models.update(dict.fromkeys(['big', 'large', 'full', 'step_383500'], 'step_383500'))
    model_name = models[model_name]
    model_path = os.path.join(model_dir, f'step_{ckpt}')
    params = sess.config

    model_path = os.path.join(model_dir, model_name, '')
    sess.state = read_ckpt(sess.state, model_path, params['devices'].shape[1])

    sess.state = sess.move_xmap(sess.state, np.zeros(params['cores_per_replica']))

    return sess
