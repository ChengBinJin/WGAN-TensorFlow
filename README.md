# WGAN-TensorFlow
This repository is a Tensorflow implementation of Martin Arjovsky's [Wasserstein GAN, arXiv:1701.07875v3](https://arxiv.org/pdf/1701.07875.pdf).

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/43870865-b795a83e-9bb4-11e8-8005-461951b3d7b7.png" width=700)
</p>
  
## Requirements
- tensorflow 1.9.0
- python 3.5.3
- numpy 1.14.2
- pillow 5.0.0
- scipy 0.19.0
- matplotlib 2.2.2

## Applied GAN Structure
1. **Generator (DCGAN)**
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/43059677-9688883e-8e88-11e8-84a7-c8f0f6afeca6.png" width=700>
</p>

2. **Critic (DCGAN)**
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/43060075-47f274d0-8e8a-11e8-88ff-3211385c7544.png" width=500>
</p>

## Generated Images
1. **MNIST**
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/43871185-659103d8-9bb6-11e8-848b-94ee5055cbe3.png" width=900>
</p>
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/43871958-1af67c1e-9bba-11e8-8a1c-4422bb19c9d0.png" width=900>
</p>

2. **CelebA**
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/43871194-79655ec2-9bb6-11e8-8b85-53fd085b0d23.png" width=900>
</p>
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/43871967-26799292-9bba-11e8-843d-08f228616d10.png" width=900>
</p>
**Note:** The results are not good as paper mentioned. We found that the Wasserstein distance can't converge well in the CelebA dataset, but it decreased in MNIST dataset. 

## Documentation
### Download Dataset
MNIST dataset will be downloaded automatically if in a specific folder there are no dataset. Use the following command to download `CelebA` dataset and copy the `CelebA' dataset on the corresponding file as introduced in **Directory Hierarchy** information.
```
python download2.py celebA
```

### Directory Hierarchy
``` 
.
│   WGAN
│   ├── src
│   │   ├── dataset.py
│   │   ├── download2.py
│   │   ├── main.py
│   │   ├── solver.py
│   │   ├── tensorflow_utils.py
│   │   ├── utils.py
│   │   └── wgan.py
│   Data
│   ├── celebA
│   └── mnist
```  
**src**: source codes of the WGAN

### Implementation Details
Implementation uses TensorFlow to train the WGAN. Same generator and critic networks are used as described in [Alec Radford's paper](https://arxiv.org/pdf/1511.06434.pdf). WGAN does not use a sigmoid function in the last layer of the critic, a log-likelihood in the cost function. Optimizer is used RMSProp instead of Adam.  

### Training WGAN
Use `main.py` to train a WGAN network. Example usage:

```
python main.py --is_train=true --dataset=[celebA|mnist]
```
 - `gpu_index`: gpu index, default: `0`
 - `batch_size`: batch size for one feed forward, default: `64`
 - `dataset`: dataset name for choice [celebA|mnist], default: `celebA`
 - `is_train`: training or inference mode, default: `False`
 - `learning_rate`: initial learning rate, default: `0.00005`
 - `num_critic`: the number of iterations of the critic per generator iteration, default: `5`
 - `z_dim`: dimension of z vector, default: `100`
 - `iters`: number of interations, default: `100000`
 - `print_freq`: print frequency for loss, default: `50`
 - `save_freq`: save frequency for model, default: `10000`
 - `sample_freq`: sample frequency for saving image, default: `200`
 - `sample_size`: sample size for check generated image quality, default: `64`
 - `load_model`: folder of save model that you wish to test, (e.g. 20180704-1736). default: `None` 

### Wasserstein Distance During Training
1. **MNIST**
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/43872167-3f0341b8-9bbb-11e8-8efa-3a9ffe6072e9.png" width=900>
</p>

2. **CelebA**
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/43872137-24c0ff7a-9bbb-11e8-8a35-5dbbba3ed743.png" width=900>
</p>

### Evaluate WGAN
Use `main.py` to evaluate a WGAN network. Example usage:

```
python main.py --is_train=false --load_model=folder/you/wish/to/test/e.g./20180704-1746
```
Please refer to the above arguments.

### Citation
```
  @misc{chengbinjin2018wgan,
    author = {Cheng-Bin Jin},
    title = {WGAN-tensorflow},
    year = {2018},
    howpublished = {\url{https://github.com/ChengBinJin/WGAN-TensorFlow}},
    note = {commit xxxxxxx}
  }
```

### Attributions/Thanks
- This project borrowed some code from [wiseodd](https://github.com/wiseodd/generative-models)
- Some readme formatting was borrowed from [Logan Engstrom](https://github.com/lengstrom/fast-style-transfer)

## License
Copyright (c) 2018 Cheng-Bin Jin. Contact me for commercial use (or rather any use that is not academic research) (email: sbkim0407@gmail.com). Free for research use, as long as proper attribution is given and this copyright notice is retained.

## Related Projects
- [Vanilla GAN](https://github.com/ChengBinJin/VanillaGAN-TensorFlow)
- [DCGAN](https://github.com/ChengBinJin/DCGAN-TensorFlow)
- [WGAN-GP](https://github.com/ChengBinJin/WGAN-GP-tensorflow)
