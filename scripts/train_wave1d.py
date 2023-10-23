import sys

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import torch.nn.functional as F

from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.datasets import load_wave1d
from neuralop.training import setup
from neuralop.training.callbacks import MGPatchingCallback, SimpleWandBLoggerCallback
from neuralop.utils import get_wandb_api_key, count_params
import os

# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./wave1d_config.yaml", config_name="default", config_folder="./config"
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="./config"),
    ]
)


class physics_wave1d(object):
    def __init__(self):
        super().__init__()

    def physics(self, u_pred):
        u = F.pad(u_pred,(1,1,1,1),"circular")
        dx = 1 / (u_pred.size(3) - 1)
        dt = 1 / (u_pred.size(2) - 1)
        utt = (u[:,0, 0:-2, 1:-1] - 2* u[:,0, 1:-1, 1:-1] + u[:,0, 2:, 1:-1]) / (dt**2)
        uxx = (u[:,0, 1:-1, 0:-2] - 2* u[:,0, 1:-1, 1:-1] + u[:,0, 1:-1, 2:]) / (dx**2)
        Du = utt - uxx
        return torch.mean(torch.abs(Du))/(torch.mean(torch.abs(uxx)+torch.abs(utt)))
            
    def __call__(self, y_pred, **kwargs):
        return self.physics(y_pred)
    
class boundary_wave1d(object):
    def __init__(self):
        super().__init__()

    def boundary(self, y_pred, y):
        return torch.mean(torch.abs(y[:,0,0,:]-y_pred[:,0,0,:])/4)

    def __call__(self, y_pred, y, **kwargs):
        return self.boundary(y_pred, y)


class combinedloss(object):
    def __init__(self, l2loss, h1loss, physics, boundary, c_h1=1, c_lp=1, c_physics=1, c_bnd =1):
        super().__init__()
        self.c_h1 = c_h1
        self.c_lp = c_lp
        self.c_physics = c_physics
        self.c_bnd = c_bnd
        self.l2loss = l2loss
        self.h1loss = h1loss
        self.physics = physics
        self.boundary = boundary

    def __call__(self, y_pred, y, **kwargs):
        return self.c_h1*self.h1loss(y_pred, y)+self.c_lp*self.l2loss(y_pred,y)+self.c_physics*self.physics(y_pred)+self.c_bnd*self.boundary(y_pred,y)



physics = physics_wave1d()

config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

# Set-up distributed communication, if using
device, is_logger = setup(config)

# Set up WandB logging
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config_name,
                config.tfno2d.n_layers,
                config.tfno2d.hidden_channels,
                config.tfno2d.n_modes_width,
                config.tfno2d.n_modes_height,
                config.tfno2d.factorization,
                config.tfno2d.rank,
                config.patching.levels,
                config.patching.padding,
            ]
        )
    wandb.init(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]

# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print config to screen
if config.verbose and is_logger:
    pipe.log()
    sys.stdout.flush()

# Loading the Darcy flow dataset
train_loader, test_loaders = load_wave1d(
    n_train=config.data.n_train,
    batch_size=config.data.batch_size,
    positional_encoding=config.data.positional_encoding,
    n_tests=config.data.n_tests,
    test_batch_size=config.data.test_batch_size,
)

model = get_model(config)
model = model.to(device)

# Use distributed data parallel
if config.distributed.use_distributed:
    model = DDP(
        model, device_ids=[device.index], output_device=device.index, static_graph=True
    )

# Log parameter count
if is_logger:
    n_params = count_params(model)

    if config.verbose:
        print(f"\nn_params: {n_params}")
        sys.stdout.flush()

    if config.wandb.log:
        to_log = {"n_params": n_params}
        if config.n_params_baseline is not None:
            to_log["n_params_baseline"] = (config.n_params_baseline,)
            to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
            to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
        wandb.log(to_log)
        wandb.watch(model)


# Create the optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)

if config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        mode="min",
    )
elif config.opt.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.opt.scheduler_T_max
    )
elif config.opt.scheduler == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    )
else:
    raise ValueError(f"Got scheduler={config.opt.scheduler}")




# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
physics = physics_wave1d()
boundary = boundary_wave1d()
combined = combinedloss(l2loss, h1loss, physics, boundary)

if config.opt.training_loss == "l2":
    train_loss = l2loss
elif config.opt.training_loss == "h1":
    train_loss = h1loss
else:
    raise ValueError(
        f'Got training_loss={config.opt.training_loss} '
        f'but expected one of ["l2", "h1"]'
    )

train_loss = combined
eval_losses = {"h1": h1loss, "l2": l2loss, "physics": physics, "boundary": boundary}

if config.verbose and is_logger:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    print(f"\n### Beginning Training...\n")
    sys.stdout.flush()
output_encoder = False
trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    device=device,
    amp_autocast=config.opt.amp_autocast,
    wandb_log=config.wandb.log,
    log_test_interval=config.wandb.log_test_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose and is_logger,
    callbacks=[
        MGPatchingCallback(levels=config.patching.levels,
                                  padding_fraction=config.patching.padding,
                                  stitching=config.patching.stitching,
                                  encoder=output_encoder),
        SimpleWandBLoggerCallback()
              ]
              )


trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

if config.wandb.log and is_logger:
    wandb.finish()
