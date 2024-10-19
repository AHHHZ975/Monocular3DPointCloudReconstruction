The code for the paper
# A High-Performance Learning-Based Framework for Monocular 3D Point Cloud Reconstruction [[Link]](https://ieeexplore.ieee.org/document/10614399)
An essential yet challenging step in the 3D reconstruction problem is to train a machine or a robot to model 3D objects. Many 3D reconstruction applications depend on real-time data processing, so computational efficiency is a fundamental requirement in such systems. Despite considerable progress in 3D reconstruction techniques in recent years, developing efficient algorithms for real-time implementation remains an open problem. The present study addresses current issues in the high-precision reconstruction of objects displayed in a single-view image with sufficiently high accuracy and computational efficiency. To this end, we propose two neural frameworks: a CNN-based autoencoder architecture called Fast-Image2Point (FI2P) and a transformer-based network called TransCNN3D. These frameworks consist of two stages: perception and construction. The perception stage addresses the understanding and extraction process of the underlying contexts and features of the image. The construction stage, on the other hand, is responsible for recovering the 3D geometry of an object by using the knowledge and contexts extracted in the perception stage. The FI2P is a simple yet powerful architecture to reconstruct 3D objects from images faster (in real-time) without losing accuracy. Then, the TransCNN3D framework provides a more accurate 3D reconstruction without losing computational efficiency.  The output of the reconstruction framework is represented in the point cloud format. The ShapeNet dataset is utilized to compare the proposed method with the existing ones in terms of computation time and accuracy. Simulations demonstrate the superior performance of the proposed strategy.

# Installation
1- We recommend using a virtual environment or a conda environment.
```
conda create -n Mono3DRecon python=3.8
```

2- Install the proper version of Pytorch library depending on your machine. For more information see the [Pytorch webpage](https://pytorch.org).
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3- Install the proper version of the Open3D library depending on your machine. For more information see the [Open3D webpage](https://www.open3d.org/docs/release/getting_started.html).
```
pip3 install open3d
```

4- Install other dependencies as follows:
```
python -m pip install -U matplotlib
pip install opencv-python
pip install pyyaml
pip install imageio
pip install neuralnet-pytorch
```

# Running the code
## Training 
To train all models, run the following command:

```
CUDA_VISIBLE_DEVICES=0 python train.py
```

## Inference
To perform the test/inference stage of all models, run the following command. Note that for running the models in the inference stage, all models should be placed in a folder with the name "PreTrainedModels" in the root directory of the project. Please see the content of the file "predict.py" to understand it better. 
```
CUDA_VISIBLE_DEVICES=0 python predict.py
```

# Dataset
Our dataset is available on the [IEEE Dataport website](https://dx.doi.org/10.21227/d9ft-0n41). To keep the footprint of this repository as small as possible, I have put data related to only one category of the whole dataset (bottle) in the "Output" folder on this repository. The user can run the commands mentioned above to train/test the 3D reconstruction models on this category of data. More data for training/evaluation can be downloaded from the link above. 

# Citation
We appreciate your interest in our research. If you want to use our work, please consider the proper citation format (the paper and the dataset) written below.
```
@article{zamani2024high,
  title={A High-Performance Learning-Based Framework for Monocular 3D Point Cloud Reconstruction},
  author={Zamani, AmirHossein and Ghaffari, Kamran and Aghdam, Amir G},
  journal={IEEE Journal of Radio Frequency Identification},
  year={2024},
  publisher={IEEE}
}

@data{d9ft-0n41-24,
doi = {10.21227/d9ft-0n41},
url = {https://dx.doi.org/10.21227/d9ft-0n41},
author = {Zamani, AmirHossein},
publisher = {IEEE Dataport},
title = {Monocular 3D Point Cloud Reconstruction Dataset (Mono3DPCL)},
year = {2024} }
```
