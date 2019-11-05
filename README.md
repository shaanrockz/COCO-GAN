# COCO_GAN

Implementation of Paper 'COCO-GAN: Generation by Parts via Conditional Coordinating'
This code is PyTorch converted version of author's TensorFlow code present here : https://github.com/hubert0527/COCO-GAN

# Prerequisite

```python
Torch
TorchVision
Numpy
Matplotlib
```

# Train the Network

To train the network, e.g. with coco-cgan, you can execute the following command:

```python
python main.py 
```
Config file can be edited according to the need !


# Dataset
"celeb_data" can be downloaded from this link : https://drive.google.com/open?id=1PceubRgNbDhTExEFSfHuB4fKsOibDKQy
This contains only 1000 randomly sampled images

For "MNIST" data, code is already present in the 'main' file under 'load_dataset'

# Reference
@inproceedings{lin2019cocogan,
  author    = {Chieh Hubert Lin and
               Chia{-}Che Chang and
               Yu{-}Sheng Chen and
               Da{-}Cheng Juan and
               Wei Wei and
               Hwann{-}Tzong Chen},
  title     = {{COCO-GAN:} Generation by Parts via Conditional Coordinating},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  year      = {2019},
}
