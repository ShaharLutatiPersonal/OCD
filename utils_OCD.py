import torch
import copy
import numpy as np
import torch.nn.functional as F
from nerf_utils.tiny_nerf import run_one_iter_of_tinynerf
from nerf_utils.nerf import cumprod_exclusive, get_minibatches, get_ray_bundle, positional_encoding
from nerf_utils.tiny_nerf import VeryTinyNerfModel

betas = torch.linspace(1e-6,1e-2,1000)
if torch.cuda.is_available():
    betas = betas.cuda()
class ConfigWrapper(object):
    """
    Wrapper dict class to avoid annoying key dict indexing like:
    `config.sample_rate` instead of `config["sample_rate"]`.
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = ConfigWrapper(**v)
            self[k] = v
      
    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def to_dict_type(self):
        return {
            key: (value if not isinstance(value, ConfigWrapper) else value.to_dict_type())
            for key, value in dict(**self).items()
        }

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def overfitting_batch_nerf(bmodel=None,weight_name='',bias_name='',
batch=None, loss_fn=None,n_iteration=10,lr=0.5e-4,verbose=False):
    base_model = copy.deepcopy(bmodel)
    param_weight = base_model.get_parameter(weight_name+'.weight')
    opt = torch.optim.Adam([
                {'params': param_weight},
            ], lr=lr)
    weight = []
    l1 = []
    for epoch in range(n_iteration):
        opt.zero_grad()
        rgb_predicted,h = run_one_iter_of_tinynerf(
                batch['height'],
                batch['width'],
                batch['focal_length'],
                batch['input'],
                batch['near_thresh'],
                batch['far_thresh'],
                batch['depth_samples_per_ray'],
                batch['encode'],
                batch['get_minibatches'],
                batch['chunksize'],
                base_model,
                batch['num_encoding_functions'],
            )
        loss = torch.nn.functional.mse_loss(rgb_predicted, batch['output'])
        if epoch == 0:
            hx = h
            hfirst = copy.deepcopy((hx.detach()))
            out = copy.deepcopy(rgb_predicted.detach())
            out_first = copy.deepcopy(out)
        loss.backward()
        opt.step()

    weight = base_model.get_parameter(weight_name+'.weight').detach()
    return weight,hfirst,out_first



def overfitting_batch_wrapper(datatype='',bmodel=None,weight_name='',bias_name='',
batch=None, loss_fn=None,n_iteration=10,lr=0.5e-4,verbose=False):
    if datatype == 'tinynerf':
        weight,hfirst,outin= overfitting_batch_nerf(
            bmodel=bmodel,weight_name=weight_name,
            bias_name=bias_name,
            batch=batch,loss_fn=loss_fn,
            n_iteration=n_iteration,
            lr=lr,
            verbose=verbose
            )
    else:
        weight,hfirst,outin= overfitting_batch(
            bmodel=bmodel,weight_name=weight_name,
            bias_name=bias_name,
            batch=batch,loss_fn=loss_fn,
            n_iteration=n_iteration,
            lr=lr,
            verbose=verbose
            )
    return weight,hfirst,outin


def overfitting_batch(bmodel=None,weight_name='',bias_name='',
batch=None, loss_fn=None,n_iteration=10,lr=0.5e-4,verbose=False):
    base_model = copy.deepcopy(bmodel)
    param_weight = base_model.get_parameter(weight_name+'.weight')
    opt = torch.optim.Adam([
                {'params': param_weight},
            ], lr=lr)
    
    for epoch in range(n_iteration):
        opt.zero_grad()
        predicted_labels,h = base_model(batch['input'].float())
        if epoch == 0:
            hx,hy = h
            hfirst = copy.deepcopy((hx.detach(),hy.detach()))
            out = copy.deepcopy(predicted_labels.detach())
        loss = loss_fn(predicted_labels, batch['output'].long())
        loss.backward()
        opt.step()
    weight = base_model.get_parameter(weight_name+'.weight').detach()
    return weight,hfirst,out

def check_ps_nerf(named_parameter='',bmodel=None,w=0,
batch=None, loss_fn=None,std=0,dopt=0):
    model = copy.deepcopy(bmodel)
    r = copy.deepcopy( model.get_parameter(named_parameter+'.weight').data)
    rgb_predicted,h = run_one_iter_of_tinynerf(
                batch['height'],
                batch['width'],
                batch['focal_length'],
                batch['input'],
                batch['near_thresh'],
                batch['far_thresh'],
                batch['depth_samples_per_ray'],
                batch['encode'],
                batch['get_minibatches'],
                batch['chunksize'],
                model,
                batch['num_encoding_functions'],
            )
    loss = loss_fn(rgb_predicted, batch['output'])
    lbase = loss.item()
    model.get_parameter(named_parameter+'.weight').data += dopt.squeeze()
    rgb_predicted,h = run_one_iter_of_tinynerf(
                batch['height'],
                batch['width'],
                batch['focal_length'],
                batch['input'],
                batch['near_thresh'],
                batch['far_thresh'],
                batch['depth_samples_per_ray'],
                batch['encode'],
                batch['get_minibatches'],
                batch['chunksize'],
                model,
                batch['num_encoding_functions'],
            )
    loss = loss_fn(rgb_predicted, batch['output'])
    loptimal = loss.item()
    model.get_parameter(named_parameter+'.weight').data = r + std*w.squeeze().to('cuda')
    rgb_predicted,h = run_one_iter_of_tinynerf(
                batch['height'],
                batch['width'],
                batch['focal_length'],
                batch['input'],
                batch['near_thresh'],
                batch['far_thresh'],
                batch['depth_samples_per_ray'],
                batch['encode'],
                batch['get_minibatches'],
                batch['chunksize'],
                model,
                batch['num_encoding_functions'],
            )
    loss = loss_fn(rgb_predicted, batch['output'])
    ldiffusion = loss.item()
    del model
    return ldiffusion,loptimal,lbase

def check_ps(named_parameter='',bmodel=None,w=0,
batch=None, loss_fn=None,std=0,dopt=0):
    model = copy.deepcopy(bmodel)
    r = copy.deepcopy( model.get_parameter(named_parameter+'.weight').data)
    predicted_labels,h = model(batch['input'])
    loss = loss_fn(predicted_labels, batch['output'].long())
    lbase = loss.item()
    model.get_parameter(named_parameter+'.weight').data += dopt.squeeze()
    predicted_labels,h = model(batch['input'])
    loss = loss_fn(predicted_labels, batch['output'].long())
    loptimal = loss.item()
    model.get_parameter(named_parameter+'.weight').data = r + std*w.squeeze().to('cuda')
    predicted_labels,h = model(batch['input'])
    loss = loss_fn(predicted_labels, batch['output'].long())
    ldiffusion = loss.item()
    del model
    return ldiffusion,loptimal,lbase

def check_ps_wrapper(isnerf=0,named_parameter='',bmodel=None,w=0,
batch=None, loss_fn=None,std=0,dopt=0):
    if isnerf:
        return check_ps_nerf(named_parameter=named_parameter,bmodel=bmodel,w=w,
batch=batch, loss_fn=loss_fn,std=std,dopt=dopt)
    else:
        return check_ps(named_parameter=named_parameter,bmodel=bmodel,w=w,
batch=batch, loss_fn=loss_fn,std=std,dopt=dopt)

def noising(x,t,padding=None):
    batch = t.shape[0]
    normalize = x.view(batch,-1).std(-1).unsqueeze(1).unsqueeze(1)
    x = x / normalize
    a,b = x.shape[1],x.shape[2]
    error = torch.randn((batch,a,b)).to(x.device)
    sigma = (1-betas).cumprod(dim=0).index_select(0, t)
    xnoisy = x*((sigma).sqrt().unsqueeze(1).unsqueeze(2)) + error*((1.0-sigma).sqrt().unsqueeze(1).unsqueeze(2))
    if len(xnoisy.shape) == 2:
        xnoisy = xnoisy.unsqueeze(0)
        error = error.unsqueeze(0)
        sigma = sigma.unsqueeze(0)
    return xnoisy,error,sigma


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a



def generalized_steps(named_parameter, numstep, x, model, bmodel, batch, loss_fn, std, padding, mat_shape, isnerf=0, **kwargs):
    with torch.no_grad():
        b = betas
        num_steps =numstep
        skip = 1000//num_steps
        x,h,outin = x
        dopt = x
        x = torch.randn_like(x)
        seq = range(0,1000,skip)
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(F.pad(xt,(padding[1][0],padding[1][1],padding[0][0],padding[0][1])), h,outin, t.float())
            et = et[:,0,padding[0][0]:padding[0][0]+mat_shape[0],padding[1][0]:padding[1][0]+mat_shape[1]]
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
        wdiff = xs[-1]
        ldiffusion,loptimal,lbase = check_ps_wrapper(isnerf=isnerf,named_parameter=named_parameter,
            bmodel=bmodel, w=wdiff.squeeze(), batch=batch,
            loss_fn=loss_fn,std=std,dopt=dopt
            )
    return ldiffusion,loptimal,lbase,wdiff

