# Implant-Detection

We provide only simulated example cases as we are not allowed to share real clinical patient data / images ...

Reference to conda yaml file and detectron2 installation instructions

## Method
![Pipeline](figures/Pipeline_Overview.png)

## Checkpoints

| Model Architecture | # Parameters | Download                           |
|:------------------:|:------------:|:----------------------------------:|
| Faster-R-CNN       | 58.8 M       | [Checkpoint](https://huggingface.co/joshua-scheuplein/Implant-Detector-Faster-R-CNN/resolve/main/Implant_Detector_Faster_R_CNN.pth) |
| Mask-R-CNN         | 61.4 M       | [Checkpoint](https://huggingface.co/joshua-scheuplein/Implant-Detector-Mask-R-CNN/resolve/main/Implant_Detector_Mask_R_CNN.pth) |

## Test Table

<table>
  <tr>
    <th> </th>
    <th colspan="2">Faster R-CNN</th>
    <th colspan="2">Mask R-CNN</th>
  </tr>
  <tr>
    <th> </th>
    <th> Bounding Boxes</th>
    <th> Keypoints</th>
    <th> Bounding Boxes</th>
    <th> Keypoints</th>
  </tr>
  <tr>
    <th> mAP </th>
    <th> 0.5</th>
    <th> 0.1</th>
    <th> 0.3</th>
    <th> 0.2</th>
  </tr>
</table>

## Inference
To perform inference run the following script [model_inference.py](model_inference.py) and is 

```bash
torchrun --nproc_per_node=4 main_dax_training.py --arch='resnet50' --flag=True'
```

## License
This repository is released under the Apache License 2.0. See [LICENSE](LICENSE) for additional details.
