from nnsight import LanguageModel

from dictionary_learning import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning.training import trainSAE

sae_device = "cpu"
activation_device = "cpu"  # these can be the same but don't have to be
lm_name = "EleutherAI/pythia-70m-deduped"  # this can be any Huggingface model

model = LanguageModel(
    lm_name,
    device_map=activation_device,
)
submodule = model.gpt_neox.layers[1].mlp  # layer 1 MLP
activation_dim = 512  # output dimension of the MLP
dictionary_size = 16 * activation_dim

# data must be an iterator that outputs strings
data = iter(
    [
        "This is some example data",
        "In real life, for training a dictionary",
        "you would need much more data than this",
    ]
    * 10_000  # synthetically increase amount of toy data since training only runs 1 epoch
)
buffer = ActivationBuffer(
    data=data,
    model=model,
    submodule=submodule,
    d_submodule=activation_dim,  # output dimension of the model component
    n_ctxs=3e2,  # you can set this higher or lower dependong on your available memory
    out_batch_size=256,
    device=activation_device,
)  # buffer will return batches of tensors of dimension = submodule's output dimension

trainer_cfg = {
    "trainer": StandardTrainer,
    "dict_class": AutoEncoder,
    "activation_dim": activation_dim,
    "dict_size": dictionary_size,
    "lr": 1e-3,
    "seed": 0,
    "wandb_name": "sae_test",
    "layer": "mlp_1",  # name of the layer in the model, used for logging
    "lm_name": lm_name,  # name of the model, used for logging
    "device": sae_device,
}

# train the sparse autoencoder (SAE)
ae = trainSAE(
    data=buffer,  # you could also use another (i.e. pytorch dataloader) here instead of buffer
    trainer_configs=[trainer_cfg],
    steps=25,  # you'll want to increase this number in practice
)
