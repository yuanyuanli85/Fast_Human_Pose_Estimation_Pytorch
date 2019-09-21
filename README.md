# Fast Human Pose Estimation Pytorch

This is an unoffical implemention for paper [Fast Human Pose Estimation, Feng Zhang, Xiatian Zhu, Mao Ye](https://arxiv.org/abs/1811.05419). 
Most of code comes from pytorch implementation for stacked hourglass network [pytorch-pose](https://github.com/bearpaw/pytorch-pose).
In this repo, we followed Fast Pose Distillation approach proposed by Fast Human Pose Estimation to improve accuracy of a lightweight network. We first trained a deep 
teacher network (stacks=8, standard convolution, **88.33**@Mpii pckh), and used it to teach a student network (stacks=2, depthwise convolution, **84.69%**@Mpii pckh).
Our experiment shows **0.7%** gain from knowledge distillation.

I benchmarked the light student model `hg_s2_b1_mobile_fpd` and got **43fps** on **i7-8700K** via **OpenVino**. Details can be found from [Fast_Stacked_Hourglass_Network_OpenVino](https://github.com/yuanyuanli85/Fast_Stacked_Hourglass_Network_OpenVino)

Please check the offical implementation by [fast-human-pose-estimation.pytorch](https://github.com/ilovepose/fast-human-pose-estimation.pytorch)

## Update at Feb 2019
* Model trained by using extra unlabeled images uploaded, `hg_s2_b1_mobile_fpd_unlabeled`  shows **0.28%** extra gain from knowledge transfered from teacher on unlabeled data. 
* The key idea is inserting unlabeled images into mpii dataset. For unlabeled samples, loss comes from difference b/w teacher and student. For labeled samples, loss is the sum of teacher-vs-student and student-vs-groundtruth.

## Results 
`hg_s8_b1`: teacher model, `hg_s2_b1_mobile`:student model, `hg_s2_b1_mobile_kd`: student model trained with FPD. `hg_s2_b1_mobile_fpd_unlabeled`: student model trained with FPD with extral unlabeled samples.

| Model|in_res |featrues| # of Weights |Head|Shoulder|	Elbow|	Wrist|	Hip	|Knee|	Ankle|	Mean| GFlops | Link|
| --- |---| ----|----------- | ----| ----| ---| ---| ---| ---| ---| ---|----| ----|
| hg_s8_b1|256|128|25.59m| 96.59| 95.35| 89.38| 84.15| 88.70| 83.98 |79.59| 88.33|28|[GoogleDrive](https://drive.google.com/open?id=1yzR8UwhklJTMMJEtjK10oyYG1tVRtkqi)
| hg_s2_b1_mobile|256|128|2.31m|95.80|  93.61| 85.50| 79.63| 86.13| 77.82| 73.62|  84.69|3.2|[GoogleDrive](https://drive.google.com/open?id=1FxTRhiw6_dS8X1jBBUw_bxHX6RoBJaJO)
| hg_s2_b1_mobile_fpd|256|128|2.31m| 95.67|94.07| 86.31| 79.68| 86.00| 79.67|75.51| 85.41|3.2|[GoogleDrive](https://drive.google.com/open?id=1zFoecNCc7alND8ODh8lg3UeRaB6_gY_V)
| hg_s2_b1_mobile_fpd_unlabeled|256|128|2.31m| 95.94|94.11| 87.18| 80.69| 87.03| 79.17|74.82| 85.69|3.2|[GoogleDrive](https://drive.google.com/open?id=1mSFD2cn8fMb1YQJPOjfU-RLVnYO1LoAl)


## Installation
1. Create a virtualenv
   ```
   virtualenv -p /usr/bin/python2.7 pose_venv
   ```
2. Clone the repository with submodule
    ```
    git clone --recursive https://github.com/yuanyuanli85/Fast_Human_Pose_Estimation_Pytorch.git
    ```
3. Install all dependencies in virtualenv
    ```
    source posevenv/bin/activate
    pip install -r requirements.txt
    ```
4. Create a symbolic link to the `images` directory of the MPII dataset:
   ```
   ln -s PATH_TO_MPII_IMAGES_DIR data/mpii/images
   ```

5. Disable cudnn for batchnorm layer to solve bug in pytorch0.4.0
    ```
    sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ./pose_venv/lib/python2.7/site-packages/torch/nn/functional.py
    ```

## Quick Demo
* Download pre-trained model[hg_s2_b1_mobile_fpd](https://drive.google.com/open?id=1zFoecNCc7alND8ODh8lg3UeRaB6_gY_V)) and save it to somewhere, i.e `checkpoint/mpii_hg_s2_b1_mobile_fpd/`
* Run demo on sample image
```buildoutcfg
python tools/mpii_demo.py -a hg -s 2 -b 1 --mobile True --checkpoint checkpoint/mpii_hg_s2_b1_mobile_fpd/model_best.pth.tar --in_res 256 --device cuda 
```     
* You will see the detected keypoints drawn on image on your screen
    
    
## Training teacher network 
* In our experiments, we used stack=8 input resolution=256 as teacher network 
```sh
python example/mpii.py -a hg --stacks 8 --blocks 1 --checkpoint checkpoint/hg_s8_b1/ 
```
* Run evaluation to get val score.
```sh
python tools/mpii.py -a hg --stacks 8 --blocks 1 --checkpoint checkpoint/hg_s8_b1/preds_best.mat 
```

## Training with Knowledge Distillation 

* Download teacher model's checkpoint or you can train from scratch. In our experiments, we used `hg_s8_b1` as teacher.  

* Train student network with knowledge distillation from teacher 
```sh
python example/mpii_kd.py -a hg --stacks 2 --blocks 1 --checkpoint checkpoint/hg_s2_b1_mobile/ mobile=True --teacher_stack 8 --teacher_checkpoint 
checkpoint/hg_s8_b1/model_best.pth.tar  
```

## Evaluation

Run evaluation to generate mat file
```sh
python example/mpii.py -a hg --stacks 2 --blocks 1 --checkpoint checkpoint/hg_s2_b1/ --resume checkpoint/hg_s2_b1/model_best.pth.tar -e
```
* `--resume_checkpoint` is the checkpoint want to evaluate

Run `tools/eval_PCKh.py` to get val score 

## Export pytorch checkpoint to onnx 
```sh
python tools/mpii_export_to_onxx.py -a hg -s 2 -b 1 --num-classes 16 --mobile True --in_res 256  --checkpoint checkpoint/model_best.pth.tar 
--out_onnx checkpoint/model_best.onnx 
```
Here 
* `--checkpoint` is the checkpoint want to export 
* `--out_onnx` is the exported onnx file


## Reference 
* OpenVino: https://github.com/opencv/dldt
* Pytorch-pose: https://github.com/yuanyuanli85/pytorch-pose
* Fast Human Pose Estimation https://github.com/yuanyuanli85/Fast_Human_Pose_Estimation_Pytorch

