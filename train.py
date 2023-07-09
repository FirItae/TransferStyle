
import argparse

from lerning import MyModel, get_args_parser
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from data_module import DataModule
import torch
from PIL import Image
import numpy as np

from pytorch_lightning.callbacks import ModelCheckpoint

# Parse command-line arguments
parser = argparse.ArgumentParser(parents=[get_args_parser()])
args = parser.parse_args()

# Create an instance of the DataModule
datamodule = DataModule(args)

# Setup the DataModule
datamodule.setup()

# Create an instance of the model
model = MyModel(args)

# Load the checkpoint from a saved file
checkpoint = torch.load(args.resume, map_location='cpu')

# Get the list of model state_dict keys and checkpoint keys
models_state_dict_keys =list(model.state_dict().keys())
#checkpoints_state_dict_keys =list(checkpoint['state_dict'].keys())

# Prepare a dictionary to store the weights to load from the checkpoint
weights_to_load = {}

# Iterate over the model state_dict keys
for idx, param_name in enumerate(models_state_dict_keys):
    # Exclude the 'criterion.empty_weight' parameter
    if not(param_name == 'criterion.empty_weight') :
        # Add the parameter to the weights_to_load dictionary
        weights_to_load[param_name] = checkpoint['state_dict'][param_name]

# Load the weights from the checkpoint into the model, allowing for strict=False (mismatched keys)
model.load_state_dict(weights_to_load, strict=False)
#model.load_state_dict(checkpoint['state_dict'], strict=False)


# Setup a checkpoint callback to save the model weights
checkpoint_callback = ModelCheckpoint(
    dirpath='/data/Transfer_Style/long',
    every_n_epochs = 2,  # Save every n epochs
)
# Create a WandbLogger for logging the training process
wandb_logger = WandbLogger(project="Transfer_Style")
# Create an instance of the Trainer
trainer = pl.Trainer(
    logger = wandb_logger,
    gpus=[0],
    accelerator="gpu",
    max_epochs= 2,
    callbacks=[checkpoint_callback]
)
# Fit the model using the Trainer and the prepared DataModule
trainer.fit(model, datamodule=datamodule)