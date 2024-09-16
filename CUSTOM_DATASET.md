## Guideline for preparing custom dataset
In this document, the process of preparing custom dataset is ellaborated.
Any dataset that provides 3D files (.obj) together with SMPL-X parameters can work.
In this document, we will describe the process with the example of THuman2.0 dataset.
---

### Dataset structure
```
$ROOT/datasets
└── THuman
   └── THuman2.0_Release
   └── THuman2.0_smplx   
   └── split_train.txt
   └── split_val.txt
   └── smplx_uv.obj  
   └── val 
      ├── img
      └── mask
      └── transform   
      └── position_map_uv_space
      └── position_map_uv_space_outer_shell_1
      └── position_map_uv_space_outer_shell_2
      └── position_map_uv_space_outer_shell_3
      └── position_map_uv_space_outer_shell_4
      └── visibility_map_uv_space
      └── visibility_map_uv_space_outer_shell_1
      └── visibility_map_uv_space_outer_shell_2
      └── visibility_map_uv_space_outer_shell_3
      └── visibility_map_uv_space_outer_shell_4         
```

---
### Render the dataset

### Create the root directory to store your custom dataset
```
mkdir $ROOT/datasets/THuman
```
### Download the raw dataset you want to process
First, download the target dataset and corresponding SMPL-X parameters.

1. Create the directory to store the raw dataset
```
cd $ROOT/datasets/THuman
mkdir THuman2.0_Release
cd THuman2.0_Release
```

2. Download the original THuman 2.0 dataset (THuman2.0_Release.zip) under `$ROOT/datasets/THuman/THuman2.0_Release`. Please complete the [request form](https://github.com/ytrock/THuman2.0-Dataset/blob/main/THUman2.0_Agreement.pdf) and send it to Yebin Liu (liuyebin@mail.tsinghua.edu.cn) and cc Tao Yu (ytrock@126.com) to request the download link.


3. Unzip the dataset.
```
unzip THuman2.0_Release.zip
```

4. Download the SMPL-X parameters (THuman2.0_smplx.tar.gz) corresponding to the original THuman 2.0 scans from [here](https://1drv.ms/u/s!Aq9xVNM_DjPG5SRFHnNVe5jQSRiv?e=QjhCAM).
Place the .tar.gz file under `$ROOT/datasets/THuman` and extract it.
```
tar -xvzf THuman2.0_smplx.tar.gz
```


5. Create the dataset split files (.txt) and place them under `$ROOT/datasets/THuman`.
You can check the dataset split files for GHG for reference.
The train split file can be downloaded [here](https://1drv.ms/t/s!Aq9xVNM_DjPG5TxZr5_kqBheaENR?e=cdkcaM). 
The test split can be downloaded [here](https://1drv.ms/t/s!Aq9xVNM_DjPG5TtjUHtwVNegpv05?e=xiUl6e).

6. Download the smplx_uv.obj from the [official SMPL-X website](https://smpl-x.is.tue.mpg.de/) and place it under `$ROOT/datasets/THuman`.

7.Render RGB images and mask images.
```
python process_dataset/render_image.py
```

### Render position maps
Please first install the NVDiffrast.
```
git clone https://github.com/NVlabs/nvdiffrast
pip install .
```


### Render visibility maps








