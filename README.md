# Implant-Detection

We provide only simulated example cases as we are not allowed to share real clinical patient data / images ...

Reference to conda yaml file and detectron2 installation instructions

## Method
![Pipeline](figures/Pipeline_Overview.png)

## Checkpoints

| Model Architecture | # Parameters | Download                           |
|--------------------|-------------:|:----------------------------------:|
| Faster-R-CNN       | 58.8 M       | [Checkpoint](https://huggingface.co/joshua-scheuplein/Implant-Detector-Faster-R-CNN/resolve/main/Implant_Detector_Faster_R_CNN.pth) |
| Mask-R-CNN         | 61.4 M       | [Checkpoint](https://huggingface.co/joshua-scheuplein/Implant-Detector-Mask-R-CNN/resolve/main/Implant_Detector_Mask_R_CNN.pth) |

## Test Table

<table>
  <tr>
    <th>Model Architecture</th>
    <th>Faster R-CNN</th>
    <th>Mask R-CNN</th>
  </tr>
  <tr>
    <th>Model Architecture</th>
    <th>Faster R-CNN</th>
    <th>Mask R-CNN</th>
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

## Inference
To perform inference run the following script [model_inference.py](model_inference.py) and is 

```bash
torchrun --nproc_per_node=4 main_dax_training.py --arch='resnet50' --flag=True'
```

## License
This repository is released under the Apache License 2.0. See [LICENSE](LICENSE) for additional details.
