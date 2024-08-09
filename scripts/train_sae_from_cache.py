""" Train a sparse autoencoder on activations from a cache directory. """

from pathlib import Path

from torch.utils.data import DataLoader

from dictionary_learning.cache_activations import get_activation_dataset_from_cache
from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning.training import trainSAE
from dictionary_learning.utils import get_device


def load_activations(
    activation_cache_dir: Path = Path("activation_cache"),
    save_dir: str = "sae_output",
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    use_wandb: bool = False,
    wandb_entity: str = "ElanaPearl",
    wandb_project: str = "sae_training",
    wandb_name: str = "tmp_run_name",  # used for wandb logging
    lm_name: str = "",  # used for wandb logging
    layer: str = "final_rs",  # used for wandb logging
    steps: int = 100_000,
    log_steps: int = 100,
):

    acts_dataset = get_activation_dataset_from_cache(activation_cache_dir)
    dataloader = DataLoader(acts_dataset, batch_size=batch_size, shuffle=True)
    print(f"Loaded dataset with {len(acts_dataset):,} tokens")

    trainer_cfg = {
        "trainer": StandardTrainer,
        "dict_class": AutoEncoder,
        "activation_dim": acts_dataset.d_model,
        "dict_size": acts_dataset.d_model * 16,
        "lr": lr,
        "seed": seed,
        "wandb_name": wandb_name,
        "layer": layer,  # used for logging
        "lm_name": lm_name,  # used for logging
        "device": get_device(),
    }

    print(f"Training with config: {trainer_cfg}")
    trainSAE(
        data=dataloader,
        trainer_configs=[trainer_cfg],
        save_dir=save_dir,
        steps=min(steps, len(dataloader)),
        log_steps=log_steps,
        use_wandb=use_wandb,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
    )


if __name__ == "__main__":
    from tap import tapify

    tapify(load_activations)
