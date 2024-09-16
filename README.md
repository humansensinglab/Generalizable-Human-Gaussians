# Generalizable Human Gaussians (GHG) for Sparse View Synthesis (ECCV 2024)

  <p align="center">
    <a href="https://youngjoongunc.github.io/"><strong>Youngjoong Kwon;</strong></a>
    ·    
    <a href="https://github.com/baolef"><strong>Baole Fang*</strong></a>
    ·
    <a href="https://lewis-lv0.github.io/"><strong>Yixing Lu*</strong></a>
    ·
    <a href="https://www.haoyed.com/"><strong>Haeoye Dong</strong></a>
    ·
    <a href="https://czhang0528.github.io/"><strong>Cheng Zhang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=3elKp9wAAAAJ&hl=en"><strong>Francisco Vicente Carrasco</strong></a>
    ·
    <a href="https://www.albertmosellamontoro.com/"><strong>Albert Mosella-Montoro</strong></a>
    ·
    <a href="https://atlantixjj.github.io/"><strong>Jianjin Xu</strong></a>
    ·
    <a href=""><strong>Shingo Takagi</strong></a>
    ·
    <a href=""><strong>Daeil Kim</strong></a>
    ·
    <a href="https://aayushp.github.io/"><strong>Aayush Prakash</strong></a>
    ·
    <a href="https://www.cs.cmu.edu/~ftorre/"><strong>Fernando de la Torre</strong></a>
  </p> 
  <p align="center" style="font-size:15px; margin-bottom:-5px !important;"><sup>*</sup>Equal contribution.</p>
  
  <p align="center">
    <a href="https://humansensinglab.github.io/Generalizable-Human-Gaussians/"><strong>Project Page</strong></a>
    |    
    <a href="https://www.youtube.com/watch?v=PyST3-dfmfI"><strong>Video</strong></a>
    |
    <a href="https://arxiv.org/abs/2407.12777"><strong>Paper</strong></a>

  </p> 


![Teaser Video](image/teaser_video_gif.gif)

**News**
* `08/26/2024` To make the comparison with our GHG easier, we provide the evaluation results in this [link](https://1drv.ms/u/s!Aq9xVNM_DjPG5RDhuwsv4XaaP62v?e=acMiPv). 
* `08/26/2024` The evaluation code and pretrained model of GHG are now released!
---

## Installation

Instructions on downloading the dataset and pretrained model weights, and installing the dependencies can be found in [INSTALL.md](INSTALL.md).

---

## Custom dataset

If you want to try GHG on your own dataset, please refer to the [CUSTOM_DATASET.md](CUSTOM_DATASET.md).

---

## Evaluation

We provide detailed information about the evaluation protocol in [PROTOCOL.md](PROTOCOL.md).
To make the comparison with our Generalizable Human Gaussians easier, we provide the evaluation results in this [link](https://1drv.ms/u/s!Aq9xVNM_DjPG5RDhuwsv4XaaP62v?e=acMiPv).

1. Please download the pretrained weights following the instructions in [INSTALL.md](INSTALL.md).
2. Generate the predictions.
    ```
    CUDA_VISIBLE_DEVICES=0 python eval.py --test_data_root datasets/THuman/val --regressor_path weights/model_gaussian.pth --inpaintor_path weights/model_inpaint.pth
    ```
   The results will be saved at `$ROOT/outputs/eval/{$exp_name}`.
3. Compute the metrics.
    ```
    python metrics/compute_metrics.py
    ``` 

## Citation

If you find this code useful for your research, please cite it using the following BibTeX entry.

```
@article{kwon2024ghg,
  title={Generalizable Human Gaussians for Sparse View Synthesis},
  author={Youngjoong Kwon, Baole Fang, Yixing Lu, Haoye Dong, Cheng Zhang, Francisco Vicente Carrasco, Albert Mosella-Montoro, Jianjin Xu, Shingo Takagi, Daeil Kim, Aayush Prakash, Fernando De la Torre},
  journal={European Conference on Computer Vision},
  year={2024}
}
```