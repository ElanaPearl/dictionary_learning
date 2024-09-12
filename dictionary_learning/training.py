"""
Training dictionaries
"""

import json
import os

import torch as t
import wandb
from tqdm import tqdm

from dictionary_learning.dictionary import AutoEncoderNew
from dictionary_learning.trainers.standard import StandardTrainer

# from .evaluation import evaluate


def trainSAE(
    data,
    trainer_config={
        "trainer": StandardTrainer,
        "dict_class": AutoEncoderNew,
        "activation_dim": 512,
        "dict_size": 64 * 512,
        "lr": 1e-3,
        "l1_penalty": 1e-1,
        "warmup_steps": 1000,
        "resample_steps": None,
        "seed": None,
        "wandb_name": "StandardTrainer",
    },
    use_wandb=False,
    wandb_entity="",
    wandb_project="",
    steps=None,
    save_steps=None,
    max_ckpts_to_keep=3,
    save_dir=None,  # use {run} to refer to wandb run
    log_steps=None,
    activations_split_by_head=False,  # set to true if data is shape [batch, pos, num_head, head_dim/resid_dim]
    transcoder=False,
    fidelity_fn=None,  # This has to be defined in the script that calls this
    eval_steps=None,
    additional_wandb_args={},
):
    """
    Train SAEs using the given trainers
    """

    trainer_cls = trainer_config["trainer"]
    trainer_config = {k: v for k, v in trainer_config.items() if k != "trainer"}
    trainer_config["steps"] = steps
    trainer = trainer_cls(**trainer_config)

    if log_steps is not None:
        if use_wandb:
            check_for_necessary_wandb_args(wandb_entity, wandb_project, log_steps)
            check_for_optional_wandb_args(trainer_config)

            wandb_config = trainer_config
            wandb_config.update(additional_wandb_args)

            wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                config=wandb_config,
                name=trainer_config["wandb_name"],
            )
            # process save_dir in light of run name
            if save_dir is not None:
                save_dir = save_dir.format(run=wandb.run.name)

    # make save dirs, export config
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        # save config
        config = {"trainer": trainer.config}
        try:
            config["buffer"] = data.config
        except:
            pass
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        if save_steps is not None:
            os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
            saved_steps = set()

    n_tokens_total = 0
    for step, act in enumerate(tqdm(data, total=steps)):
        if steps is not None and step >= steps:
            print("Stopped training because reached max specified steps")
            break

        # logging
        if log_steps is not None and step % log_steps == 0:
            log = {}
            with t.no_grad():
                if not transcoder:
                    act, act_hat, f, losslog = trainer.loss(
                        act, step=step, logging=True
                    )  # act is x

                    # L0: avg number of non-zero features
                    n_nonzero_per_example = (f != 0).float().sum(dim=-1)
                    l0 = n_nonzero_per_example.mean().item()
                    # L0_norm: avg pct of non-zero features per example
                    l0_norm = (
                        n_nonzero_per_example / trainer_config["dict_size"]
                    ).mean().item() * 100

                    # fraction of variance explained
                    total_variance = t.var(act, dim=0).sum()
                    residual_variance = t.var(act - act_hat, dim=0).sum()
                    frac_variance_explained = 1 - residual_variance / total_variance
                    log["frac_variance_explained"] = frac_variance_explained.item()
                else:  # transcoder
                    x, x_hat, f, losslog = trainer.loss(
                        act, step=step, logging=True
                    )  # act is x, y

                    # L0
                    l0 = (f != 0).float().sum(dim=-1).mean().item()

                    # fraction of variance explained
                    # TODO: adapt for transcoder
                    # total_variance = t.var(x, dim=0).sum()
                    # residual_variance = t.var(x - x_hat, dim=0).sum()
                    # frac_variance_explained = (1 - residual_variance / total_variance)
                    # log[f'{trainer_name}/frac_variance_explained'] = frac_variance_explained.item()

                # check if losslog has NaN and stop
                if losslog["loss"] != losslog["loss"]:
                    print("Oh no, NaN loss!!")
                    breakpoint()

                # log parameters from training
                log.update(losslog)
                log["l0"] = l0
                log["l0_pct_nonzero"] = l0_norm
                trainer_log = trainer.get_logging_parameters()
                trainer_log.update(trainer.get_extra_logging_parameters())
                for name, value in trainer_log.items():
                    log[name] = value

                if fidelity_fn is not None and step % eval_steps == 0:
                    # Note, we assume function takes activations and returns a dict
                    # TODO: make a fidelity_fn super class that follows this API
                    fidelity = fidelity_fn(sae_model=trainer.ae)
                    for k, v in fidelity.items():
                        log[k] = v

                # add in the mean and std of act and act_hat
                log["act_mean"] = act.mean().item()
                log["act_std"] = act.std().item()
                log["reconstruction_mean"] = act_hat.mean().item()
                log["reconstruction_std"] = act_hat.std(dim=1).mean().item()
                log["tokens"] = n_tokens_total

                # TODO get this to work
                # metrics = evaluate(
                #     trainer.ae,
                #     data,
                #     device=trainer.device
                # )
                # log.update(
                #     {f'trainer{i}/{k}' : v for k, v in metrics.items()}
                # )
            if use_wandb:
                wandb.log(log, step=step)

        # saving
        if save_steps is not None and step % save_steps == 0:
            t.save(
                trainer.ae.state_dict(),
                os.path.join(save_dir, "checkpoints", f"ae_{step}.pt"),
            )
            # add step to the set saved_steps
            saved_steps.add(step)

            # if there are more than the max files, delete the one with the smallest step
            if len(saved_steps) > max_ckpts_to_keep:
                min_step = min(saved_steps)
                saved_steps.remove(min_step)
                os.remove(os.path.join(save_dir, "checkpoints", f"ae_{min_step}.pt"))

        # training
        trainer.update(step, act)

        # update n_tokens_total
        n_tokens_total += act.shape[0]

    # save final SAEs
    if save_dir is not None:
        t.save(trainer.ae.state_dict(), os.path.join(save_dir, "ae.pt"))

    # End the wandb run
    if log_steps is not None and use_wandb:
        wandb.finish()


def check_for_necessary_wandb_args(wandb_entity, wandb_project, log_steps):
    """
    Check if necessary arguments are present for logging to wandb.

    Raises:
        ValueError: If any necessary arguments are missing.
    """
    necessary_args = {
        "wandb_entity": wandb_entity,
        "wandb_project": wandb_project,
        "log_steps": log_steps,
    }

    missing_args = [arg for arg, value in necessary_args.items() if not value]

    if missing_args:
        raise ValueError(
            "In order to log your run to wandb, you must specify the following arguments:\n"
            + "\n".join(f"* {arg}" for arg in missing_args)
        )


def check_for_optional_wandb_args(trainer_cfg):
    """
    Check if helpful but optional arguments are present for logging to wandb.

    Prints a warning if any optional arguments are missing.
    """
    optional_args = ["wandb_name", "layer", "lm_name"]
    missing_args = [arg for arg in optional_args if arg not in trainer_cfg]

    if missing_args:
        print(
            "Warning: You are missing the following optional arguments from trainer_cfg:\n"
            + "\n".join(f"* {arg}" for arg in missing_args)
            + "\nIt will still log your run, but these are useful for tracking purposes."
        )
