import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from utils_OCD import overfitting_batch_wrapper,noising,generalized_steps
import torch.utils.tensorboard as tb
from ema import EMAHelper
import torchvision
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class NoiseModel(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg11(pretrained=True)
        self.vgg = vgg

    def forward(self, x):
        out = self.vgg(x)
        return out
noise_model = NoiseModel().to(device)
def vgg_encode(x):
    with torch.no_grad():
        return noise_model(x.unsqueeze(0).permute(0,3,1,2).contiguous())
def train(args, config, optimizer, optimizer_scale,
        device, diffusion_model, scale_model,
        model,  train_loader, padding, mat_shape,
        ema_helper, tb_logger, loss_fn,
        opt_error_loss):

    epochs = config.training.epochs
    weight_name = config.model.weight_name
    grad_accum = config.training.grad_accum
    grad_clip = config.training.grad_clip
    checkpoint_path = config.checkpoint.checkpoint_path
    n_checkpoint = config.checkpoint.n_checkpoint
    diffusion_num_steps = config.diffusion.diffusion_num_steps
    lr_overfitting = config.overfitting.lr_overfitting
    n_overfitting = config.overfitting.n_overfitting
    step = 0
    dmodel_original_weight = deepcopy(model.get_parameter(weight_name+'.weight'))
    if args.precompute_all == 1:
        print('precomputation of overfitting to save time starts')
        ws,hs,outs = [],[],[]
        for idx, batch in enumerate(train_loader):
            optimizer_scale.zero_grad()
            batch['input'] = batch['input'].to(device)
            batch['output'] = batch['output'].to(device)
            # Overfitting encapsulation #
            weight,hfirst,outin= overfitting_batch_wrapper(
            datatype=args.datatype,
            bmodel=model,weight_name=weight_name,
            bias_name=weight_name,
            batch=batch,loss_fn=opt_error_loss,
            n_iteration=n_overfitting,
            lr=lr_overfitting,
            verbose=False
            )
            ws.append(deepcopy(weight.detach().cpu()))
            hs.append(deepcopy(hfirst))
            outs.append(deepcopy(outin.detach().cpu()))

        print('precomputation finished')
    print('Start Training')
    for epoch in range(epochs):
        avg_loss = 0
        count = 0
        optimizer.zero_grad()
        difflosslogger = 0
        optimizer_scale.zero_grad()
        for idx, batch in enumerate(train_loader):
            optimizer_scale.zero_grad()
            batch['input'] = batch['input'].to(device)
            batch['output'] = batch['output'].to(device)
            # Overfitting encapsulation #
            if args.precompute_all:
                weight,hfirst,outin = ws[idx].to(device),hs[idx],outs[idx].to(device)
            else:
                weight,hfirst,outin= overfitting_batch_wrapper(
                datatype=args.datatype,
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
            if args.datatype == 'tinynerf':
                encoding_out = vgg_encode(outin)
            else:
                encoding_out = outin
            estimated_error = diffusion_model(
                F.pad(weight_noisy,(padding[1][0],padding[1][1],padding[0][0],padding[0][1])),
                hfirst,
                encoding_out,
                t.float()
                )
            scale = scale_model(hfirst,encoding_out) # estimate scale
            estimated_error = estimated_error[:,0,padding[0][0]:padding[0][0]+mat_shape[0],padding[1][0]:padding[1][0]+mat_shape[1]] #remove padding
            ascale = diff_weight.view(-1).std() # calculate optimal scale
            lscale = 10*torch.log10(((scale.squeeze()-ascale).square())/(ascale.square()+1e-12) + 1e-8) # scale loss 
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
            print(f'epoch {epoch+1} save checkpoints: model_checkpoint_epoch{epoch}_step{step}_data{args.datatype}, scale_model_checkpoint_epoch{epoch}_loss{step}_data{args.datatype}')
            torch.save(ema_helper.state_dict(),checkpoint_path+f'model_checkpoint_epoch{epoch}_step{step}_data{args.datatype}.pt')
            torch.save(scale_model.state_dict(),checkpoint_path+f'scale_model_checkpoint_epoch{epoch}_loss{step}_data{args.datatype}.pt')
    return diffusion_model,scale_model
