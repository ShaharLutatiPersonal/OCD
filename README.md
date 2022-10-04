## OCD: LEARNING TO OVERFIT WITH CONDITIONAL DIFFUSION MODELS
Official PyTorch Implementation

- THIS REPO WILL BE UPDATE IN THE NEXT DAYS -

# Prerequisite
1. pytorch (verified on 1.12.0)
2. torchvision
3. numpy
4. scipy
5. copy
6. tensorboard

# Models
in module diffusion_ocd:
1. Model - Diffusion network 
2. Model_Scale - Scale estimation network

# Training
In order to use the training, fill in the blank paths-
module_path = ""  -> path to desired pretrained model (f_\theta)
tb_path = "" -> path to tensorboard log
data_path_train = "" -> path to training data (should be a pytorch Dataset class)
data_path_test = "" -> path to testing data (should be a pytorch Dataset class)
checkpoint_path = "" -> path to save checkpoints
weight_name = '' -> name of the parameter to overfit

# Special notes before running

1. Currently the code supports only 2D weights (i.e. weights of Linear units)
2. Since the overfitting is 1 sample at a time, a gradient accumulation is performed over the diffusion training. In future release a batchified version will be released. It is crucial to have gradient accumulation for the batch-normalization to work properly.
3. Magic constants are magical, but you can change it if you would like (learning rate, etc...)
4. The basemodel should output the following:
predicted_labels,(hx,hy)
where predicted_labels is the base model output, and the tuple (hx,hy) contains the latent input to the layer and latent output of the layer.
5. dim_in, dim_out in Model_Scale stands for dim of latent into the layer and latent dim outside.
6. dim_output in Model stands for the output dimension of the basemodel.


