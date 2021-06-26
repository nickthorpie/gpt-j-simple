import shutil
from gpt_j_simple.src._utils import default_params

def train_step(network, data):
    inputs = {
        "obs": data[:, :, :-1],
        "target": data[:, :, 1:],
    }

    loss, last_loss = network.train(inputs)

    return np.array(loss).mean(), np.array(last_loss).mean()


def eval_step(network, data):
    inputs = {
        "obs": data[:, :-1],
        "target": data[:, 1:],
    }

    out = network.eval(inputs)
    loss = out["loss"]

    return np.array(loss).mean()


def save(network,
         step,
         model_dir,
         project_name=None,
         mp=None, aux=None,
         keep_n=2,
         delete_old=True):

    if project_name is not None:
        model_dir = os.path.join(model_dir, project_name)
    if aux is None:
        aux = {}

    try:
        with open(f"{model_dir}/meta.json", "r") as f:
            meta = json.load(f)
    except:
        # create metadata file
        with open(f"{model_dir}/meta.json", "w") as f:
            json.dump({
                "step": 0,
                "checkpoints": [],
                "aux": {}
            }, f)

    # do sharded checkpoint writing
    start = time.time()
    res = []
    for shard_id in range(mp):
        write_ckpt(network.state, f"{model_dir}/step_{step}/", shard_id)

    print(f"Wrote checkpoint in {time.time() - start:.06}s")

    with open(f"{model_dir}/meta.json", "r") as f:
        meta = json.load(f)

    meta["step"] = step
    meta["checkpoints"].append(step)
    all_aux = meta.get("aux", {})

    while len(meta["checkpoints"]) > keep_n:
        ckpt_to_delete = meta["checkpoints"].pop(0)

        try:
            del all_aux[str(ckpt_to_delete)]
        except:
            print(f"failed to delete the aux state for {step}")

        if delete_old:
            print(f"deleting checkpoint {ckpt_to_delete}")
            if 'gs://' in model_dir:
                raise Exception("Can't manage buckets yet.")
                # bucket, path = model_dir.split('/')[-2:]
                # for blob in client.list_blobs(bucket,
                #                               prefix=f"{path}/step_{ckpt_to_delete}/"):
                #     # print(f"deleting {blob.name}")
                #     assert path in blob.name
                #     blob.delete()
            else:
                shutil.rmtree(f'{model_dir}/step_{ckpt_to_delete}/')
        else:
            print(f"keeping checkpoint {ckpt_to_delete}")

    all_aux[step] = aux
    meta["aux"] = all_aux

    with open(f"{model_dir}/meta.json", "w") as f:
        json.dump(meta, f)


def train(train_set,
          model_dir='models',
          project_name='gptj_slim',
          ckpt='step_383500',
          finetune=True,
          params={}
          ):
    """
    A method to finetune our dataset locally.
    Psuedocode:


    This is adapted from device_train.py.

    args:
    """
    params = default_params(params)
    ##resolve ckpt input to be just the int:
    if isinstance(ckpt, (int)):
        None
    elif isinstance(ckpt, str):
        if 'step' in ckpt.split('_')[0]:
            ckpt = int(ckpt.split('_')[1])
    elif ckpt is None:
        None

    ##init optimizer
    opt = optax.chain(
        optax.scale(1 / gradient_accumulation_steps),
        clip_by_global_norm(1),
        optax.scale_by_adam(),
        additive_weight_decay(weight_decay),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(warmup_steps, anneal_steps,
                                                   lr, end_lr))
    )
    ## Check if session has been created
    if sess is not None:
        msg = "Sorry we don't support loading in a session from a variable. Yet."
        msg+= "Provide model_dir,project_name,ckpt instead"
        raise Exception(msg)
        # sess.config['optimizer']=opt

    params = default_params()params
    devices = _init_colab_tpu_jax()
    num_devices = len(devices)

    if num_devices < cores_per_replica:
        msg = f"each shard needs a separate device, but device count "
        msg+= f"({num_devices}) < shard count ({cores_per_replica})"
        msg+= f"Please reshard using **TODO:MAKE QUICK RESHARD FUNC**"
        raise ValueError(msg)

    mesh_shape = (num_devices // cores_per_replica, cores_per_replica)
    devices = np.array(devices).reshape(mesh_shape)
    print(f"jax devices: {num_devices}")

    train_loader = None

    model_dir = os.path.join(model_dir, project_name)
    meta_path = f"{model_dir}/meta.json"

    train_loader = None
    if ckpt is 'latest':
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            ckpt = meta["checkpoints"][-1]
            ckpt_path = f"{model_dir}/step_{ckpt}/"
            step = ckpt_step
            train_loader = meta['aux'][str(ckpt)].get("train_loader", None)
        except NotFound:
            msg = f"No checkpoint to load at {model_dir}. Training from scratch. "
            msg += f"If a checkpoint exists with no 'meta.json', use ckpt='step_NO. "
            msg += f"Starting from scratch."
            print(msg)
    elif ckpt is None:
        msg = "No checkpoint requested. Starting from scratch. Use ckpt='step_NO'"
        msg += "to load a checkpoint"
    else:
        ckpt_path = os.path.join(model_dir, f"step_{ckpt}")
        if not os.path.exists(ckpt_path):
            msg = f'didnt find ckpt = step_{ckpt}. Try passing ckpt=None to start from '
            msg += f'scratch, or ckpt = "latest" to look for one'
            raise Exception(msg)

    print("setting up datasets")

    # train_dataset = TFRecordNewInputs(f"data/{train_set}",
    #                                   batch_size=(
    #                                       gradient_accumulation_steps,
    #                                       per_replica_batch * num_devices // cores_per_replica),
    #                                   sample_size=params['seq'],
    #                                   restore_state=train_loader)

    global_val_batch = per_replica_batch * num_devices // cores_per_replica
    val_sets = {}

    # for k, v in params['val_set'].items():
    #     val_sets[k] = TFRecordNewInputs(f"data/{v}",
    #                                     batch_size=(global_val_batch,),
    #                                     sample_size=seq)

    # tok/sec metrics
    windows_per_step = gradient_accumulation_steps * (per_replica_batch * num_devices // cores_per_replica)
    tokens_per_step = params['seq'] * windows_per_step

    with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
        print("initializing jax sess")
        network = CausalTransformer(params)

        if ckpt:
            print(f"loading ckpt_{ckpt}")
            if finetune:
                # get the scheduler step stored in the just-initialized optimizer
                # should be zero
                init_sched_state = network.state["opt_state"][-1]
            start = time.time
            network.state = read_ckpt(network.state, ckpt_path, devices.shape[1])
            if finetune:
                # overwrite the loaded scheduler step with zeros
                # this makes fine-tuning use the lr schedule in
                network.state["opt_state"][-1] = init_sched_state

        print('compiling train fn')
        start = time.time()
        train_step(network, train_dataset.get_samples())
        step += 1
        print(f"Train fn compiled in {time.time() - start:.06}s")

        print('compiling eval fn')
        start = time.time()
        for val_set in val_sets.values():
            eval_step(network, val_set.get_samples())
            val_set.reset()
        print(f"Eval fn compiled in {time.time() - start:.06}s")

        wandb.init(project='mesh-transformer-jax', name=params["name"], config=params)

        while True:
            if (step % ckpt_every == 1) or step == total_steps:
                print(f"saving a checkpoint for step {step}")
                save(network,
                     step, model_dir,
                     project_name=None,
                     mp=cores_per_replica,
                     aux={"train_loader": train_dataset.get_state()},
                     keep_n=2,
                     delete_old=True)

                if step == total_steps:
                    print("training completed!")
                    exit()

            if step % val_every == 1:  # 1 because we've already taken a step to compile train fn
                for name, val_set in val_sets.items():
                    val_loss = []
                    for i, _ in tqdm(zip(val_set.sample_once(), range(val_batches)),
                                     desc=f"validation for step {step}, set {name}",
                                     total=val_batches):
                        val_loss.append(eval_step(network, i))
                    val_set.reset()

                    val_loss = np.array(val_loss).mean()
                    print(f"validation loss for step {step}, set {name}: {val_loss}")

                    wandb.log({f'val/loss_{name}': float(val_loss)}, step)

            start = time.time()
            loss, last_loss = train_step(network, train_dataset.get_samples())
            step += 1

            steps_per_sec = 1 / (time.time() - start)
            tokens_per_sec = tokens_per_step * steps_per_sec

            wandb.log({'train/loss': loss, 'train/last_loss': last_loss, 'train/steps_per_sec': steps_per_sec,
                       'train/tokens_per_sec': tokens_per_sec}, step)