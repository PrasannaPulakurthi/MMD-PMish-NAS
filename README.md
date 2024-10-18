## Enhancing GANs with MMD Neural Architecture Search, PMish Activation Function, and Adaptive Rank Decomposition [[Website]](https://prasannapulakurthi.github.io/mmdpmishnas/)
This repository contains code for our 2024 IEEE ACCESS Journal paper "**Enhancing GANs with MMD Neural Architecture Search, PMish Activation Function and Adaptive Rank Decomposition,**" authored by [Prasanna Reddy Pulakurthi](https://www.prasannapulakurthi.com/), [Mahsa Mozaffari](https://mahsamozaffari.com/), [Sohail Dianat](https://www.rit.edu/directory/sadeee-sohail-dianat), [Jamison Heard](https://www.rit.edu/directory/jrheee-jamison-heard), [Raghuveer Rao](https://ieeexplore.ieee.org/author/37281258600), and [Majid Rabbani](https://www.rit.edu/directory/mxreee-majid-rabbani).

# PMish Activation Function

This is an implementation of the **PMish Activation** function using PyTorch. It combines the `Tanh` and `Softplus` functions with a learnable parameter `beta`.

## Code

```python
import torch
import torch.nn as nn

class PMishActivation(nn.Module): 
    def __init__(self): 
        super(PMishActivation, self).__init__() 
        self.beta = nn.Parameter(torch.ones(1).type(torch.cuda.FloatTensor))  # Learnable parameter beta
        self.tanh_fn = nn.Tanh()
        self.softplus_fn = nn.Softplus()
        
    def forward(self, x): 
        return x * self.tanh_fn((1 / self.beta) * self.softplus_fn(self.beta * x))
```

## Getting Started
### Installation
1. Clone this repository.

    ~~~
    git clone https://github.com/PrasannaPulakurthi/MMD-PMish-NAS.git
    cd MMD-PMish-NAS
    ~~~
   
2. Install requirements using Python 3.9.

    ~~~
    conda create -n mmd-nas python=3.9
    conda activate mmd-nas
    pip install -r requirements.txt
    ~~~
    
2. Install Pytorch1 and Tensorflow2 with CUDA.

    ~~~
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
    ~~~
    To install other Pytroch versions compatible with your CUDA. [Install Pytorch](https://pytorch.org/get-started/previous-versions/)
   
    [Install Tensorflow](https://www.tensorflow.org/install/pip#windows-native)


## Instructions for Testing, Training, Searching, and Compressing the Model.
### Preparing necessary files

Files can be found in [Google Drive](https://drive.google.com/drive/folders/1o7DZ2R9B1yvHgVjUqhA9IpioCJ4ZYGMV?usp=sharing).

1. Download the pre-trained models to ./exps from the exps folder found [here](https://drive.google.com/drive/folders/1IinAvKxnc2Vb6-nNKV5tfYcWPwiZ1QiK?usp=drive_link). 
    
2. Download the pre-calculated statistics to ./fid_stat for calculating the FID from [here](https://drive.google.com/drive/folders/1W9_z_rhs9fZ_rs8iUn_y8DBr4FyNJWLP?usp=drive_link).

### Testing
1. Download the trained generative models from [here](https://drive.google.com/drive/folders/1wqjsFDP1Trj8dZVcFAl_nk5YOxaBtnys?usp=drive_link) to ./exps/train/pmishact_large_cifar10_xx/Model

    ~~~
    mkdir -p exps/train/pmishact_large_cifar10_xx/Model
    ~~~
   
2. To test the trained model, run the command found in scripts/test_arch_cifar10.sh
   
    ~~~
    python MGPU_test_arch.py --random_seed 33333 --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint train/pmishact_large_cifar10_33333_2024_04_18_19_59_27 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --exp_name test/pmishact_large_cifar10 --act pmishact
    ~~~

### Training
1. Train the weights of the generative model with the searched architecture (the architecture is saved in ./exps/arch_cifar10/Genotypes/latest_G.npy). Run the command found in scripts/train_arch_cifar10_large.sh
   
    ~~~
    python MGPU_train_arch.py --gpu_ids 0 --num_workers 1 --gen_bs 128 --dis_bs 128 --dataset cifar10 --bottom_width 4 --img_size 32 --max_epoch_G 500 --n_critic 1 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --df_dim 512 --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --val_freq 5 --num_eval_imgs 50000 --exp_name arch_train_cifar10_large
    ~~~

### Searching the Architecture

1. To use AdversarialNAS to search for the best architecture, run the command found in scripts/search_arch_cifar10.sh
   
    ~~~
    python MGPU_search_arch.py --gpu_ids 0 --gen_bs 128 --dis_bs 128 --dataset cifar10 --bottom_width 4 --img_size 32 --max_epoch_G 25 --arch search_both_cifar10 --latent_dim 120 --gf_dim 160 --df_dim 80 --g_spectral_norm False --d_spectral_norm True --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --n_critic 5 --val_freq 5 --derive_freq 1 --derive_per_epoch 16 --draw_arch False --exp_name search/bs120-dim160 --num_workers 8 --gumbel_softmax True
    ~~~
    
### Compression
1. Apply the ARD to find the best ranks for each layer.
   
   To find the FID score for each layer and a candidate rank, run the Python file with the following command. 
   ~~~
   python scripts/findrank.py
   ~~~
   Run the ARD.ipynb jupyter notebook file to find the optimal ranks.
   
    ## Optimal Ranks found by ARD
    | PF   | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Layer 5 | Layer 6 | Layer 7 | Layer 8 | Layer 9 | Layer 10 | Layer 11 | Layer 12 | Layer 13 | l1      | l2      | l3      |
    |------|---------|---------|---------|---------|---------|---------|---------|---------|---------|----------|----------|----------|----------|---------|---------|---------|
    | 1/1  | 128     | 128     | 256     | 128     | 256     | 128     | 128     | 128     | 512     | 128      | 128      | 128      | 768      | nc      | 2       | 2       |
    | 1/5  | 128     | 128     | 768     | 128     | 768     | 128     | 128     | 256     | 512     | 256      | 128      | 128      | 768      | nc      | 2       | 2       |
    | 1/10 | 128     | 128     | 768     | 128     | 768     | 128     | 128     | 256     | 768     | 256      | 768      | 768      | nc       | nc      | 2       | 2       |
    | 1/15 | 128     | 128     | 768     | 128     | 768     | 128     | 128     | 512     | nc      | 256      | 768      | 768      | nc       | nc      | 2       | 2       |
    | 1/20 | 128     | 128     | 768     | 128     | nc      | 128     | 128     | nc      | nc      | 512      | 768      | 768      | nc       | nc      | 2       | 2       |

2. Compress and Finetune all the Convolutional Layers according to ARD.

    ~~~
    python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --freeze_before_compressed --freeze_layers l1 l2 l3 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --df_dim 512 --num_eval_imgs 50000 --checkpoint train/pmishact_large_cifar10_33333_2024_04_18_19_59_27 --exp_name compress/rp_1 --val_freq 5 --gen_bs 128 --dis_bs 128 --beta1 0.0 --beta2 0.9 --byrank --rank 128 128 768 128 768 128 128 512 nc 256 768 768 nc nc 2 2 --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 cell2.c0.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 cell3.c0.ops.0.op.1 cell3.c1.ops.0.op.1 cell3.c2.ops.0.op.1 cell3.c3.ops.0.op.1 cell3.c4.ops.0.op.1 l1 l2 l3 --max_epoch_G 300 --act pmishact
    ~~~
       
3. To Test the compressed network, download the compressed model from [here](https://drive.google.com/drive/folders/1E94LwSQ4ah69W2HMhy6y1f34vEaEExrx?usp=drive_link) to ./exps/Compress/cifar10_small_1by15_xx/Model

    ~~~
    python MGPU_test_cpcompress.py --gpu_ids 0 --num_workers 1 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint Compress/cifar10_small_1by15_2024_05_05_02_38_59 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name Compress/test_cifar10_small  --byrank
    ~~~

## Acknowledgement
Codebase from [AdversarialNAS](https://github.com/chengaopro/AdversarialNAS), [MMD-AdversarialNAS](https://github.com/PrasannaPulakurthi/MMD-AdversarialNAS), and [Tensorly](https://github.com/tensorly/tensorly).
