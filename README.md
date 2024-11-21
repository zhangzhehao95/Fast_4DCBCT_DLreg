# Fast motion-compensated reconstruction for 4D-CBCT using deep learning-based artifact reduction and groupwise registration
Previous work has that deep learning (DL)-enhanced 4D cone beam computed tomography (4D-CBCT) images improve motion modeling and subsequent motion-compensated (MoCo) reconstruction for 4D-CBCT. However, building the motion model at treatment time via conventional deformable image registration (DIR) methods is not temporally feasible.

This work aims to improve the efficiency of 4D-CBCT MoCo reconstruction using DL-based groupwise registration for the rapid generation of a motion model prior to treatment. 

Authored by: Zhehao Zhang, Yao Hao, Xiyao Jin, Deshan Yang, Ulugbek S. Kamilov, Geoffrey D. Hugo

## Citation



## To get started

### Artifact-reduced CBCT images
The motion models were estimated from artifact-reduction 4D-CBCTs, generated using a pre-trained artifact-reduction network with initial FDK reconstructions. For more details and the code implementation of the artifact-reduction network, please refer to our previous paper.

Zhang, Z, Liu, J, Yang, D, Kamilov, US, Hugo, GD. Deep learning-based motion compensation for four-dimensional cone-beam computed tomography (4D-CBCT) reconstruction. Med Phys. 2022; 1- 13. https://doi.org/10.1002/mp.16103

### DL-based groupwise registration
* Two separate groupwise registration models were trained and evaluated, a patient-specific model and a population model. Here we provide the Python code for registration models.

* The key difference between the two models lies in their training strategies. The patient-specific model employes a one-shot learning strategy, where only a single set of artifact-reduced 4D-CBCTs to be registered is as training data. In contrast, the population model is trained beforehand using multiple sets of pre-collected 4D images. Once trained, the population model can directly infer a DVF for new patient’s 4D-CBCT images, making it substantially faster than training or updating a patient-specific model on the fly.

#### Prerequisites 
* python 3.8.16
* torch 1.9.0
* numpy 1.24.2 2.0.0
* SimpleITK 2.2.1

You may install the requirements using `pip install -r requirements.txt`

#### Run
* Our pretrined weights for the population model can be found [here](https://drive.google.com/drive/folders/1SQpLSTw4Xaug8S4rkXMHdbLWYEgWMCkX?usp=sharing). Save them under `experiments\4DCT\population_model\saved_model`.

* One set of artifact-reduced SPARE challenge data are also [provided](https://drive.google.com/drive/folders/1SQpLSTw4Xaug8S4rkXMHdbLWYEgWMCkX?usp=sharing) as example. Once moving it into `data\` folder, you can utilize the pre-trained population model to generate the DVFs by
  ```
  python code/main.py -c configs/population_model.py -d data/ -r experiments/
  ```
  The results will be saved under `experiments/4DCT/population_model/test_results`, in which the interphase DVFs are named 'dvf2phasei_j', representing the DVFs between phase _j_ and phase _i_.

* For patient-specific model, the model needs to be trained from scratch for each new to-be-registered images by
  ```
  python code/main.py -c configs/oneshot_model.py -d data/ -r experiments/
  ```

#### Dataset
Two publicly available datasets are used in this project for performance evaluation. You can get the [DIR-Lab 4DCT dataset](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html) for assessing registration accuracy and the [SPARE challenge dataset](https://image-x.sydney.edu.au/spare-challenge/) for evaluating 4D-CBCT reconstruction quality.

### MoCo reconstruction
The initial FDK and MoCo reconstruction are implemented using the Reconstruction Toolkit (RTK) v2.1.0.

## Contact
The code is provided to support reproducible research. If there is any unknown error or you need further help, please feel free to contact us at zhehao.zhang@wustl.edu.

