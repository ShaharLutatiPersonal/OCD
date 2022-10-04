from pickle import FALSE
import torch
import torch.functional as F
from math import floor as floor
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from copy import deepcopy
import os
import numpy as np
from diffusion_ocd import Model,Model_Scale
from utils_OCD import overfitting_batch,noising,generalized_steps
import torch.utils.tensorboard as tb
from ema import EMAHelper

####################################### Parameters & Initializations #####################################
torch.manual_seed(123456789)
module_path = "" #path to desired pretrained model
tb_path = "" # path to tensorboard log
tb_logger = tb.SummaryWriter(log_dir=tb_path)
data_path_train = ""# path to training data
data_path_test = ""# path to testing data
checkpoint_path = "" # path to save checkpoint
weight_name = '' # name of the parameter to overfit
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 10000000
grad_clip = 0.1
lr = 1e-4 # learning rate for the diffusion model & scale estimation model
lr_overfitting = 3e-4 # learning rate for the overfitting procedure
n_overfitting = 10 # number of overfitting steps
n_diff_step_eval = 10 # number of diffusion steps for evaluation (ddim scheduling)
n_checkpoint = 10 # save checkpoint each n_checkpoint epochs
diffusion_num_steps = 1000 # number of steps in the diffusion process
model = torch.load(module_path)
model = model.to(device)
diffusion_model = Model().cuda()
loss_fn = torch.nn.MSELoss()
scale_model = Model_Scale().cuda()
grad_accum = 128 # Accumulate results from "mini batches" since we overfit for batch of 1 sample.
# Split randomly the train data set to validation and train
train_samples_num, validation_samples_num = None, None
train_dataset = torch.load(data_path_train)
test_dataset = torch.load(data_path_test)
train_set, val_set = torch.utils.data.random_split(
        train_dataset, [train_samples_num, validation_samples_num])
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)
opt_error_loss = torch.nn.MSELoss() # Change according to desired objective
dmodel_original_weight = deepcopy(model.get_parameter(weight_name+'.weight'))
optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=lr)
optimizer_scale= torch.optim.Adam(scale_model.parameters(), lr=lr)
ema_helper = EMAHelper(mu=0.9999)
ema_helper.register(diffusion_model)
step = 0
################################################# Check if weight is OK ##########################
mat_shape = dmodel_original_weight.shape
assert len(mat_shape) == 2, "Weight to overfit should be a matrix !"
padding = []
for s in mat_shape:
    if s == 128:
        padding.append([0,0])
    elif s<128:
        rem = 128-s
        if (rem % 2) == 0:
            padding.append([rem//2,rem//2])
        else:
            padding.append([rem//2 + 1,rem//2])
    elif (s % 2) == 1:
        padding.append([2,1])
    else:
        padding.append([0,0])
#################################################################################################

########################################### Train Phase #####################################

################################################################################################

for epoch in range(epochs):
    avg_loss = 0
    count = 0
    optimizer.zero_grad()
    difflosslogger = 0
    optimizer_scale.zero_grad()
    for idx, data in enumerate(train_loader):
        optimizer_scale.zero_grad()
        train_x, train_label = data
        batch = {'input':train_x.to(device),'output':train_label.to(device)}
        # Overfitting encapsulation #
        weight,hfirst,outin= overfitting_batch(
        bmodel=model,weight_name=weight_name,
        bias_name=weight_name,
        batch=batch,loss_fn=opt_error_loss,
        n_iteration=n_overfitting,
        lr=lr_overfitting,
        verbose=False
        )
        diff_weight = weight - dmodel_original_weight #calculate optimal weight difference from baseline
        t = torch.randint(low=0, high=diffusion_num_steps, size=(1,)
                ).to(device) #Sample random timestamp
        weight_noisy,error,sigma = noising(diff_weight,t)
        estimated_error = diffusion_model(
            F.pad(weight_noisy,(padding[1][0],padding[1][1],padding[0][0],padding[0][1])),
            hfirst,
            outin,
            t.float()
            )
        scale = scale_model(hfirst) # estimate scale
        estimated_error = estimated_error[:,0,padding[0][0]:padding[0][0]+mat_shape[0],padding[1][0]:padding[1][0]+mat_shape[1]] #remove padding
        ascale = diff_weight.view(-1).std() # calculate optimal scale
        lscale = 10*torch.log10((scale.squeeze()-ascale).square()/ascale.square()) # scale loss 
        lossdiff = (loss_fn(estimated_error , error))/ grad_accum  # diffusion loss
        difflosslogger += lossdiff.item()
        tb_logger.add_scalar("loss_scale", lscale.item(), global_step=step)
        step += 1
        count += 1
        lossdiff.backward()
        lscale.backward()
        ############# Gradient accumulation for diffusion steps #################
        if ((idx + 1) % grad_accum == 0) or (idx + 1 == len(train_loader)):
            
            tb_logger.add_scalar("loss_diff", difflosslogger, global_step=step//grad_accum) 
            difflosslogger = 0
            torch.nn.utils.clip_grad_norm_(
                            diffusion_model.parameters(), grad_clip,error_if_nonfinite=True
                        )
            optimizer.step()
            ema_helper.update(diffusion_model)
            optimizer.zero_grad()
        ############################################################################    
        torch.nn.utils.clip_grad_norm_(
                        scale_model.parameters(), grad_clip,error_if_nonfinite=True
                    )
        optimizer_scale.step()
        optimizer_scale.zero_grad()
    if ((epoch + 1) % n_checkpoint == 0) or (epoch + 1 == epochs):
        torch.save(ema_helper.state_dict(),checkpoint_path+f'model_checkpoint_epoch{epoch}_step{step}.pt')
        torch.save(scale_model.state_dict(),checkpoint_path+f'scale_model_checkpoint_epoch{epoch}_loss{step}.pt')
#################################################################################################

########################################### Test Phase #####################################

################################################################################################
diffusion_model.eval()
scale_model.eval()
ldiff,lopt,lbaseline = 0,0,0
for idx, data in enumerate(val_loader):
    test_x, test_label = data
    batch = {'input':test_x.to(device),'output':test_label.to(device)}
    # Overfitting encapsulation #
    weight,hfirst,outin= overfitting_batch(
    bmodel=model,weight_name=weight_name,
    bias_name=weight_name,
    batch=batch,loss_fn=opt_error_loss,
    n_iteration=n_overfitting,
    lr=lr_overfitting,
    verbose=False
    )
    diff_weight = weight - dmodel_original_weight
    with torch.no_grad():
        std = scale_model(hfirst)
    ldiffusion, loptimal, lbase, wdiff = generalized_steps(
        named_parameter=weight_name, numstep=n_diff_step_eval,
        x=(diff_weight,hfirst,outin), model=diffusion_model,
        bmodel=model, batch=batch, loss_fn=opt_error_loss,
        std=std, padding=padding,
        mat_shape=mat_shape
        )
    ldiff += ldiffusion
    lopt += loptimal
    lbaseline += lbase
    print(f"Baseline loss {lbaseline/(idx+1)}, Overfitted loss {lopt/(idx+1)}, Diffusion loss {ldiff/(idx+1)}")
