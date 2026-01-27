# Implant-Detection

We provide only simulated example cases as we are not allowed to share real clinical patient data / images ...

Reference to conda yaml file and detectron2 installation instructions

## Method
![DAX_Method](figures/DAX_Method_Figure.png)

https://huggingface.co/joshua-scheuplein/Implant-Detector-Mask-R-CNN/resolve/main/Implant_Detector_Mask_R_CNN.pth

https://huggingface.co/joshua-scheuplein/Implant-Detector-Faster-R-CNN/resolve/main/Implant_Detector_Faster_R_CNN.pth

## Checkpoints

| Model           | # Parameters  | # mAP@0.5 | # mAR@0.5 | Download                           |
|-----------------|--------------:|----------:|----------:|:----------------------------------:|
| Faster-R-CNN    | 11.2 M        | [Checkpoint](https://huggingface.co/joshua-scheuplein/Implant-Detector-Faster-R-CNN/resolve/main/Implant_Detector_Faster_R_CNN.pth) |
| Mask-R-CNN      | 23.5 M        | [Checkpoint](https://huggingface.co/joshua-scheuplein/Implant-Detector-Mask-R-CNN/resolve/main/Implant_Detector_Mask_R_CNN.pth) |

## Test Table

<table>
  <tr>
    <th>Model</th>
    <th># Parameters</th>
    <th>mAP@0.5</th>
    <th>mAR@0.5</th>
    <th>Download</th>
  </tr>
  <tr>
    <td rowspan="2">R-CNN Family</td>
    <td>11.2 M</td>
    <td>0.72</td>
    <td>0.65</td>
    <td>
      <a href="https://huggingface.co/...">Faster R-CNN</a>
    </td>
  </tr>
  <tr>
    <td>23.5 M</td>
    <td>0.75</td>
    <td>0.68</td>
    <td>
      <a href="https://huggingface.co/...">Mask R-CNN</a>
    </td>
  </tr>
</table>

## Evaluation
In order to use the already pretrained DAX backbones for feature extraction in a custom downstream task, the script [load_checkpoints.py](code/load_checkpoints.py) demonstrates how the provided checkpoints can be loaded and used with only a few lines of code. However, one should always ensure to apply the same image preprocessing that has been used during model pretraining, such that the distribution of the input data aligns with the checkpoint weights. The detailed implementation of all preprocessing steps can be found in the script [utils.py](code/utils.py) and is 

```bash
torchrun --nproc_per_node=4 main_dax_training.py --arch='resnet50' --norm_last_layer=True --use_bn_in_head=True --use_fp16=False --clip_grad=0 --global_crops_scale 0.14 1.0 --local_crops_scale 0.05 0.14 --local_crops_number=6 --dataset='DAX-Dataset-{version}' --data_path='path/to/dataset' --augmentation='v2' --output_dir='path/to/output/directory' --num_workers=10 --seed=0 --weight_decay=1e-6 --weight_decay_end=1e-6 --batch_size_per_gpu=128 --epochs=200 --freeze_last_layer=1 --saveckp_freq=1 --warmup_teacher_temp=0.04 --teacher_temp=0.07 --warmup_teacher_temp_epochs=25 --lr=0.3 --warmup_epochs=10 --min_lr=0.0048 --optimizer='lars' --momentum_teacher=0.996 --out_dim=60000 --job_ID='DAX_Training_Job_xxx' --use_wandb='False' --pretrained_weights='path/to/checkpoint' --subtract_lowpass='False' --azure='True'
```

## License
This repository is released under the Apache License 2.0. See [LICENSE](LICENSE) for additional details.
