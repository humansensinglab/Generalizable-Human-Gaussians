### Set up the python environment

```
# clone the project
git clone https://github.com/humansensinglab/Generalizable-Human-Gaussians.git

# make sure that the pytorch cuda is consistent with the system cuda
conda env create --file environment.yml
conda activate ghg
ROOT=/path/to/Generalizable-Human-Gaussians
cd $ROOT

# install requirement for the differentiable Gaussian rasterizer
git clone --recurse-submodules https://github.com/ashawkey/diff-gaussian-rasterization.git
cd diff-gaussian-rasterization
python setup.py install
```

### Set up datasets

#### THuman 2.0 dataset

1. We are sharing the rendered dataset to those who already acquired access from the original THuman dataset authors. 
Please first request the access to the dataset following the instructions [here](https://github.com/ytrock/THuman2.0-Dataset).
Then please email [here](mintchocchoc@yahoo.com) with the title "Request for GHG dataset", and attach the screenshot of confirmation from THuman authors.

2. Create the "datasets" directory and place the downloaded files in the following structure:
```
$ROOT/datasets
└── THuman
   └── val
      ├── img
      └── mask
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

### Set up model weights
We provide the pretrained models [here](https://1drv.ms/f/s!Aq9xVNM_DjPG5QD0W3TddUkp5aUT?e=sHwO4N). Place the model weights in the following structure:
```
$ROOT/weights
├── model_gaussian.pth
└── model_inpaint.pth
```