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

<table border="1" style="border-collapse: collapse; width:100%;">
  <!-- Define column widths -->
  <colgroup>
    <col style="width:20%;">
    <col style="width:20%;">
    <col style="width:20%;">
    <col style="width:20%;">
    <col style="width:20%;">
  </colgroup>

  <!-- Header rows -->
  <tr>
    <th rowspan="2" style="text-align:center;">Metric</th>
    <th colspan="2" style="text-align:center;">Faster R-CNN</th>
    <th colspan="2" style="text-align:center;">Mask R-CNN</th>
  </tr>
  <tr>
    <th style="text-align:center;">BBs</th>
    <th style="text-align:center;">KPs</th>
    <th style="text-align:center;">BBs</th>
    <th style="text-align:center;">KPs</th>
  </tr>

  <!-- Data rows -->
  <tr>
    <td style="text-align:left;"><b>mAP@0.50:0.95</b></td>
    <td style="text-align:center;">60.9 &plusmn; 3.0</td>
    <td style="text-align:center;">53.8 &plusmn; 4.1</td>
    <td style="text-align:center;">64.7 &plusmn; 2.6</td>
    <td style="text-align:center;">58.0 &plusmn; 4.4</td>
  </tr>
  <tr>
    <td style="text-align:left;">mAP@0.75</td>
    <td style="text-align:center;">72.6 &plusmn; 3.2</td>
    <td style="text-align:center;">56.7 &plusmn; 4.1</td>
    <td style="text-align:center;">78.4 &plusmn; 3.3</td>
    <td style="text-align:center;">62.0 &plusmn; 5.4</td>
  </tr>
  <tr>
    <td style="text-align:left;">mAP@0.50</td>
    <td style="text-align:center;">86.4 &plusmn; 2.7</td>
    <td style="text-align:center;">77.5 &plusmn; 4.9</td>
    <td style="text-align:center;">87.0 &plusmn; 2.3</td>
    <td style="text-align:center;">79.5 &plusmn; 4.2</td>
  </tr>
  <tr>
    <td style="text-align:left;">mAR@0.50:0.95</td>
    <td style="text-align:center;">65.4 &plusmn; 3.0</td>
    <td style="text-align:center;">61.9 &plusmn; 4.4</td>
    <td style="text-align:center;">70.4 &plusmn; 2.2</td>
    <td style="text-align:center;">67.9 &plusmn; 3.7</td>
  </tr>
  <tr>
    <td style="text-align:left;">mAR@0.75</td>
    <td style="text-align:center;">76.5 &plusmn; 3.2</td>
    <td style="text-align:center;">65.6 &plusmn; 4.4</td>
    <td style="text-align:center;">82.6 &plusmn; 2.5</td>
    <td style="text-align:center;">71.8 &plusmn; 4.1</td>
  </tr>
  <tr>
    <td style="text-align:left;">mAR@0.50</td>
    <td style="text-align:center;">88.0 &plusmn; 2.6</td>
    <td style="text-align:center;">80.6 &plusmn; 4.5</td>
    <td style="text-align:center;">88.8 &plusmn; 2.3</td>
    <td style="text-align:center;">83.6 &plusmn; 3.3</td>
  </tr>
</table>

## Inference
To perform inference run the following script [model_inference.py](model_inference.py) and is 

```bash
torchrun --nproc_per_node=4 main_dax_training.py --arch='resnet50' --flag=True'
```

## License
This repository is released under the Apache License 2.0. See [LICENSE](LICENSE) for additional details.
