## OCD Learning to Overfit with Conditional Diffusion Models🪐<br><sub>Official PyTorch Implementation</sub>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ocd-learning-to-overfit-with-conditional/few-shot-text-classification-on-average-on)](https://paperswithcode.com/sota/few-shot-text-classification-on-average-on?p=ocd-learning-to-overfit-with-conditional)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ocd-learning-to-overfit-with-conditional/image-classification-on-tiny-imagenet-1)](https://paperswithcode.com/sota/image-classification-on-tiny-imagenet-1?p=ocd-learning-to-overfit-with-conditional)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ocd-learning-to-overfit-with-conditional/speech-separation-on-libri5mix)](https://paperswithcode.com/sota/speech-separation-on-libri5mix?p=ocd-learning-to-overfit-with-conditional)

### [Paper](https://arxiv.org/abs/2210.00471)
We present a dynamic model in which the weights are conditioned on an input sample x and are learned to match those that would be obtained by finetuning a base model on x and its label y. This mapping between an input sample and network weights is shown to be approximated by a linear transformation of the sample distribution, which suggests that a denoising diffusion model can be suitable for this task. The diffusion model we therefore employ focuses on modifying a single layer of the base model and is conditioned on the input, activations, and output of this layer. Our experiments demonstrate the wide applicability of the method for **Image Classification**, **3D Reconstruction**, **Tabular Data**, **Few-Shot NLP**, and **Speech Separation**.

## Updates 
07.10.22 - tinyNeRF is online! 💥💥💥<br />
11.10.22 - LeNet5 (MNIST) is online! :100: :100: :100:
## Setup
1. Clone the repo to your local machine.
2. Make sure you install the requirements.
## Special notes
1. You can use either the example of tinyNerf as in the code (and also train it by yourself) or take Lenet5 model.
Please see the Lenet5 model, for full explantion to how export the latent input and output for the selected layer.
The output should be:
predicted_labels, h = base_model(); <br />where h is I(x) as in the paper.
***It is very important to detach() the latent variables in the forward pass of the base model, and also use "deepcopy" function from copy***
2. For tinyNerf it is suggested to use the flag --precompute_all, for Lenet5 not.<br />
3. Specific configs are in config folder, although the generic config as in the paper will work too. <br />
The specific configs are optimizied for small footprint to allow low-end devices to run the model.<br />
4. Make sure you correctly change the name of the selected layer if other network is employed.<br />
5. The diffusion process will work once the training objective (over the normalized diffusion model) reaches a plateau of 5E-4.
## Examples:
**1. for training nerf-OCD** <br />
```
python run_func_OCD.py -e 0
```
**2. for evaluating nerf** <br />
```
python run_func_OCD.py -e 1 -t 0 -pd ./checkpoints/model_ocd_tinynerf.pt -ps ./checkpoints/scale_model_tinynerf.pt
```
**3. for training lenet5-OCD**<br />
```
python run_func_OCD.py -e 0 -pb ./checkpoints/checkpoint_lenet5.pth -pc ./configs/train_mnist.json -pdtr ./data/mnist -pdts ./data/mnist -dt mnist -prc 0
```
**4. for evaluating lenet5-OCD** - First you need to train the model(!) <br />
```
python run_func_OCD.py -e 1 -t 0 -pb ./checkpoints/checkpoint_lenet5.pth -pc ./configs/train_mnist.json -pdtr ./data/mnist -pdts ./data/mnist -dt mnist -prc 0 -pd ./checkpoints/model_ocd_mnist.pt -ps ./checkpoints/scale_model_mnist.pt
```
## Acknowledgments
The tinyNeRF code base is from (https://github.com/krrish94/nerf-pytorch) Krishna Murthy's repo. <br />
There are adjustments in order to export the latent variables required for the diffusion process condition. <br />
The diffusion model is largely adopted from (https://github.com/ermongroup/ddim)  Jiaming Song's repo.
