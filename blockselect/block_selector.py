import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import KernelDensity

class block_entropy_calc(nn.Module):
    def __init__(self, network,input_generator,loss_func,monte_carlo_num = 1000,batch_size = 1000,state_dict = None, device = "cuda"):
        super().__init__()
        self.net = network
        self.state_dict = state_dict
        self.net = self.net.to(device)
        self.gen_inputs = input_generator
        self.loss = loss_func
        self.names = [n for n,p in self.net.named_modules() if hasattr(p,'weight')]
        self.monte_carlo_num = monte_carlo_num
        self.batch_size = batch_size
        self.H_max = [-100 for i in self.names]
        self.epoch = 0
        self.testing_vec = np.linspace(0,100,100)[:, np.newaxis]


#         data_transforms = {
#     'train': transforms.Compose([
#         transforms.ToTensor(),
#     ]),
#     'val': transforms.Compose([
#         transforms.ToTensor(),
#     ]),
#     'test': transforms.Compose([
#         transforms.ToTensor(),
#     ])
# }

# data_dir = 'tiny-imagenet-200/'
# num_workers = {
#     'train' : 100,
#     'val'   : 0,
#     'test'  : 0
# }
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x])
#                   for x in ['train', 'val','test']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=100,
#                                              shuffle=True, num_workers=num_workers[x])
#               for x in ['train', 'val', 'test']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}



    def forward(self,specific_blk = None):
        with torch.no_grad():
            if specific_blk is not None:
                self.names = [self.names[specific_blk]]
            for i,block in enumerate(self.names):
                self.init_net()
                for epoch in range(self.monte_carlo_num):
                    self.epoch = epoch
                    self.change_block_weights(block)
                    loss = []
                    for b in range(self.batch_size):
                        gen_input,c = next(self.gen_inputs) # This part should be changed for other architecture
                        output = self.net(gen_input) # This part should be changed for other architecture
                        loss.append(self.loss(output,c).item())
                    self.H_max[i] = self.calculate_Hmax(loss,self.testing_vec,self.H_max[i])
                print('block {} , {}/{}'.format(block,i,len(self.names)-1))
        return np.array(self.H_max)/np.max(self.H_max)
    
    def fix_2names(self):
        new_names = []
        for n in self.names:
            shape = eval('self.net.{}.weight.data.shape'.format(n))
            if len(shape)>1:
                new_names.append(n)
        self.names = new_names

    def init_net(self):
        if self.state_dict is None:
            for p in self.net.parameters():
                if hasattr(p,'weight'):
                    nn.init.kaiming_normal_(p.data)
        else:
            self.net.load_my_state_dict(self.state_dict)
    def clipped_log(self,x):
        w = []
        for xx in x:
            if xx == 0:
                w.append(0)
            else:
                w.append(np.log10(xx))
        return np.array(w)

    def change_block_weights(self,block):
        for n,p in self.net.named_modules():
            if n == block:
                with torch.no_grad():
                    if len(p.weight.data.shape)>1:
                        nn.init.kaiming_normal_(p.weight.data)
                    else:
                        p.weight.data = nn.Parameter(torch.randn(p.weight.data.shape).cuda())

    def recursive_variance(self,block_class,data):
        for d in data:
            block_class.push(d)
        


    def calculate_Hmax(self,out,testing_vec,H_max):
        out = np.array(out)
        out = out.reshape(-1,1)
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(out)
        kde_est = kde.score_samples(testing_vec)
        H_curr = -np.sum(np.exp(kde_est)*kde_est)
        if H_curr > H_max:
            H_max = H_curr
        return H_max

    def fix_names(self,names):
        for ii,n in enumerate(names):
            x = n.split('.')
            for iii,word in enumerate(x):
                try:
                    a = int(word)
                except ValueError:
                    if iii != (len(x) -1):
                        x[iii] = '{}.'.format(word)
                    continue
                x[iii] = '[{}]'.format(word)
                if iii != (len(x)-1):
                    x[iii] += '.'
            names[ii] = ''.join(x)
            names[ii] = names[ii].replace('.[','[')
        return names



import math

class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())