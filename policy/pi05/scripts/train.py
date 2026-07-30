import dataclasses
import functools
import json
import logging
import math
import os
import pathlib
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.casm as casm
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.performance as _performance
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    enabled: bool = True,
):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name, settings=wandb.Settings(disable_code=True))
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            settings=wandb.Settings(disable_code=True),
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def _cast_floating_param_to_float32(param):
    value = getattr(param, "value", None)
    if value is not None and hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating):
        return param.replace(value.astype(jnp.float32))
    return param


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(
        config.optimizer,
        config.lr_schedule,
        weight_decay_mask=None,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16 unless float32 is requested for newer GPU compiler compatibility.
        if config.pytorch_training_precision == "bfloat16":
            params = nnx_utils.state_map(
                params,
                config.freeze_filter,
                lambda p: p.replace(p.value.astype(jnp.bfloat16)),
            )
        elif config.pytorch_training_precision == "float32":
            params = params.map(lambda _, p: _cast_floating_param_to_float32(p))

        return training_utils.TrainState(
            step=0,
            microstep=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
    ):
        if getattr(config.model, "casm_mode", "none") in casm.VISUAL_PHASE_GATE_MODES:
            chunked_loss, aux = model.compute_loss(rng, observation, actions, train=True, return_aux=True)
        else:
            chunked_loss, aux = model.compute_loss(rng, observation, actions, train=True), {}
        return jnp.mean(chunked_loss), aux

    train_rng = jax.random.fold_in(rng, state.microstep)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    next_microstep = state.microstep + 1
    optimizer_update = next_microstep % config.gradient_accumulation_steps == 0
    step_increment = jnp.asarray(optimizer_update, dtype=jnp.asarray(state.step).dtype)
    new_state = dataclasses.replace(
        state,
        step=state.step + step_increment,
        microstep=next_microstep,
        params=new_params,
        opt_state=new_opt_state,
    )
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: jnp.where(
                    optimizer_update,
                    state.ema_decay * old + (1 - state.ema_decay) * new,
                    old,
                ),
                state.ema_params,
                new_params,
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        "learning_rate": config.lr_schedule.create()(state.step),
    }
    info.update(aux)
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    compilation_cache = os.environ.get("JAX_COMPILATION_CACHE_DIR", "~/.cache/jax")
    jax.config.update("jax_compilation_cache_dir", str(epath.Path(compilation_cache).expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
        params_only=config.params_only_checkpoint,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    data_iter = iter(data_loader)
    data_wait_started = time.perf_counter()
    batch = next(data_iter)
    data_wait_ms = (time.perf_counter() - data_wait_started) * 1000
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    start_microstep = int(train_state.microstep)
    accumulation_steps = config.gradient_accumulation_steps
    if start_microstep != start_step * accumulation_steps:
        raise ValueError(f"checkpoint is not on an optimizer boundary: step={start_step}, microstep={start_microstep}")
    total_microsteps = config.num_train_steps * accumulation_steps
    pbar = tqdm.tqdm(total=config.num_train_steps, initial=start_step, dynamic_ncols=True)

    performance_receipt = None
    performance_path = os.environ.get("PARALLELVLA_PERFORMANCE_RECEIPT")
    compile_s = None
    if performance_path:
        performance_receipt = _performance.TimingReceipt(
            pathlib.Path(performance_path),
            warmup_steps=int(os.environ.get("PARALLELVLA_PERFORMANCE_WARMUP_STEPS", "5")),
            metadata={
                "framework": "jax",
                "mode": "end-to-end",
                "config_name": config.name,
                "batch_size": config.batch_size * accumulation_steps,
                "microbatch_size": config.batch_size,
                "gradient_accumulation_steps": accumulation_steps,
                "images_per_sample": 3,
                "num_workers": config.num_workers,
                "prefetch_factor": config.prefetch_factor,
                "persistent_workers": config.persistent_workers,
                "pin_memory": config.pin_memory,
                "seed": config.seed,
                "code_commit": os.environ.get("PARALLELVLA_CODE_COMMIT"),
                "dataset_revision": os.environ.get("PARALLELVLA_DATASET_REVISION"),
                "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
                "xla_memory_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
            },
        )
        compile_started = time.perf_counter()
        with sharding.set_mesh(mesh):
            ptrain_step = ptrain_step.lower(train_rng, train_state, batch).compile()
        compile_s = time.perf_counter() - compile_started

    infos = []
    first_step_s = None
    try:
        for microstep in range(start_microstep, total_microsteps):
            microstep_started = time.perf_counter()
            compute_started = time.perf_counter()
            with sharding.set_mesh(mesh):
                train_state, info = ptrain_step(train_rng, train_state, batch)
            if performance_receipt is not None:
                jax.block_until_ready((train_state, info))
            compute_ms = (time.perf_counter() - compute_started) * 1000

            completed_microsteps = microstep + 1
            optimizer_update = completed_microsteps % accumulation_steps == 0
            optimizer_step = completed_microsteps // accumulation_steps

            logging_started = time.perf_counter()
            infos.append(info)
            if optimizer_update:
                pbar.update(1)
            if optimizer_update and optimizer_step % config.log_interval == 0:
                stacked_infos = common_utils.stack_forest(infos)
                reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
                info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
                pbar.write(f"Step {optimizer_step}: {info_str}")
                wandb.log(reduced_info, step=optimizer_step)
                infos = []
            logging_ms = (time.perf_counter() - logging_started) * 1000

            checkpoint_started = time.perf_counter()
            checkpoint_saved = optimizer_update and (
                optimizer_step % config.save_interval == 0 or optimizer_step == config.num_train_steps
            )
            if checkpoint_saved:
                _checkpoints.save_state(
                    checkpoint_manager,
                    train_state,
                    data_loader,
                    optimizer_step,
                    params_only=config.params_only_checkpoint,
                )
                if performance_receipt is not None:
                    checkpoint_manager.wait_until_finished()
            checkpoint_ms = (time.perf_counter() - checkpoint_started) * 1000

            if performance_receipt is not None:
                host_info = jax.device_get(info)
                metrics = {name: float(value) for name, value in host_info.items()}
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError(f"non-finite metrics at microstep {completed_microsteps}: {metrics}")
                microstep_total_ms = (time.perf_counter() - microstep_started) * 1000
                if first_step_s is None:
                    first_step_s = microstep_total_ms / 1000
                performance_receipt.append(
                    {
                        "step": optimizer_step,
                        "microstep": completed_microsteps,
                        "optimizer_update": optimizer_update,
                        "data_wait_ms": data_wait_ms,
                        "compute_ms": compute_ms,
                        "logging_ms": logging_ms,
                        "checkpoint_ms": checkpoint_ms,
                        "checkpoint_saved": checkpoint_saved,
                        "step_total_ms": microstep_total_ms,
                        **metrics,
                    }
                )

            next_data_wait_ms = 0.0
            if completed_microsteps < total_microsteps:
                data_wait_started = time.perf_counter()
                batch = next(data_iter)
                next_data_wait_ms = (time.perf_counter() - data_wait_started) * 1000
            data_wait_ms = next_data_wait_ms
    finally:
        pbar.close()
        if performance_receipt is not None:
            summary_path = performance_receipt.close()
            summary = json.loads(summary_path.read_text())
            summary["compile_s"] = compile_s
            summary["first_compiled_step_s"] = first_step_s
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
