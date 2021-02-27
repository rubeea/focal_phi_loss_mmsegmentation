<div align="center">
  <img src="acunet.png" width="600"/>
</div>
<br />



## Focal Phi Loss with ACU-Net

This repository holds the source code for Focal Phi Loss, a novel loss function for highly imbalanced datasets applied to Power line detection with Auxiliary Classifier U-Net (ACU-Net), by Rabeea Jaffari, Manzoor Ahmed Hashmani and Constantino Carlos Reyes-Aldasoro.


### Prerequisites
Python >= 3.5  
PyTorch == 1.6, tested on CUDA 10.2. The models were trained and evaluated on PyTorch 1.6. When testing with other versions, the results (metrics) are slightly different.
CUDA, to compile the NMS code  
Other dependencies described in requirements.txt  
The versions described here were the lowest the code was tested with. Therefore, it may also work in other earlier versions, but it is not guaranteed (e.g., the code might run, but with different outputs).

### Install
The code in this repo is built using the mmsegmentation framework. For more information on the mmsegmentation framework see:  
https://github.com/open-mmlab/mmsegmentation  
https://mmsegmentation.readthedocs.io/en/latest/  

### Datasets
The two benchmark Power line (PL) datasets used in this research are:
1. Mendeley PL dataset available at:
2. Power line dataset of urban scenes (PLDU) available at: 

The train/val splits of these datasets can be found at: [dataset_files](../dataset_files)

## License

This project is released under the [Apache 2.0 license](LICENSE).

