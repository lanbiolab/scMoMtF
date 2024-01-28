# scMoMtF

scMoMtF is an interpretable multitask framework for comprehensive analyzing single-cell multi-omics data. This repository is dedicated to providing the code used to perform all tasks in the scMoMtF paper.

scMoMtF can simultaneously solve multiple key tasks of single-cell multi-omics data:

- dimension reduction
- cell classification
- data simulation

# Running environment

python==3.9.12

pandas==1.2.4

numpy==1.22.4

torch==1.13.1

scikit-learn==1.1.1

scanpy==1.9.5

episcanpy==0.4.0

ezdict==1.0.0

# Download the example datasets

SNARE-seq dataset is provided as an example in this repository and the remaining datasets in the paper can be obtained from our [Google Drive](https://drive.google.com/drive/folders/1jNxGvTbaiIxncWMzwDrMlGpipCL1NYyb?usp=sharing).

In addition, the downloaded datasets should be placed in the data folder, consistent with the SNARE seq dataset:

```
  scMoMtF      
     ├─data
     │  ├─CITE
     │  ├─PBMC
     │  ├─SHARE
     │  └─SNARE
     │      ├─ATAC.h5ad
     │      └─RNA.h5ad
     │          
     ├─model
     │  
   . . .      
```

# Training

If you want to obtain the training results of scMoMtF on SNARE-seq, please run the following file:

```
run main.py
```

# Result

After training, the trained model is available in the **trained_model** folder and the outputs of each task are available in the **output** folder.



