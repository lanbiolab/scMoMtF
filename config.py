from ezdict import EZDict

args_rc = EZDict({
    "seed": 1024,
    "query": True,
    "classification": True,
    "dim_reduce": True,
    "simulation": True,
    "batch_size": 128,
    "epochs": 30,
    "lr": 0.02,
    "z_dim": 64,
    "hidden_1": 150,
    "latent_dim": 64,
    "hidden_2": 150,
    'n_clusters': 11,
    "train_loss_weight": None,
    "encoder":
        {
            "input": 2000,
            "hiddens": [256, 128],
            "output": 64,
            "use_biases": [True, True, True],
            "dropouts": [0, 0, 0],
            "activations": ["relu", "relu", "relu"],
            "use_batch_norms": [True, True, True],
            "use_layer_norms": [False, False, False],
            "is_binary_input": False,
        },
    "discriminator":
        {
            "input": 2006,
            "hiddens": [64],
            "output": 1,
            "use_biases": [True, True],
            "dropouts": [0, 0],
            "activations": ["relu", "sigmoid"],
            "use_batch_norms": [True, True],
            "use_layer_norms": [False, False],
        }
})
