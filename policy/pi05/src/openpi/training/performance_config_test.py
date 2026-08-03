from openpi.training import config


def test_putcab_pytorch_and_jax_casm_configs_match_training_semantics():
    pytorch = config.get_config("pi05_putcab_casm_visual_phase_gate_pytorch_full")
    jax = config.get_config("pi05_putcab_casm_visual_phase_gate_jax_full")

    assert pytorch.model == jax.model
    assert pytorch.data == jax.data
    assert pytorch.batch_size == jax.batch_size == 16
    assert pytorch.num_train_steps == jax.num_train_steps == 20_000
    assert pytorch.save_interval == jax.save_interval == 2_000
    assert pytorch.keep_period == jax.keep_period == 20_000
    assert pytorch.lr_schedule == jax.lr_schedule
    assert pytorch.optimizer == jax.optimizer
    assert pytorch.ema_decay is None
    assert jax.ema_decay is None
    assert pytorch.pytorch_training_precision == "float32"
    assert pytorch.pytorch_compute_precision == "bfloat16"
    assert pytorch.num_workers == 2
    assert pytorch.prefetch_factor == 2
    assert pytorch.persistent_workers is True
    assert pytorch.pin_memory is True
    assert pytorch.pytorch_gradient_checkpointing_scope == "vision"
    assert pytorch.pytorch_compile_mode == "default"


def test_putcab_pytorch_and_jax_baseline_configs_match_training_semantics():
    pytorch = config.get_config("pi05_putcab_pytorch_matched_full")
    jax = config.get_config("pi05_putcab_jax_matched_full")

    assert pytorch.model == jax.model
    assert pytorch.data == jax.data
    assert pytorch.batch_size == jax.batch_size == 16
    assert pytorch.num_train_steps == jax.num_train_steps == 20_000
    assert pytorch.save_interval == jax.save_interval == 2_000
    assert pytorch.lr_schedule == jax.lr_schedule
    assert pytorch.optimizer == jax.optimizer
    assert pytorch.ema_decay is jax.ema_decay is None
    assert pytorch.pytorch_training_precision == "float32"
    assert pytorch.pytorch_compute_precision == "bfloat16"
    assert pytorch.num_workers == 2
    assert pytorch.prefetch_factor == 2
    assert pytorch.persistent_workers is True
    assert pytorch.pin_memory is True
    assert pytorch.pytorch_gradient_checkpointing_scope == "vision"
    assert pytorch.pytorch_compile_mode == "default"
