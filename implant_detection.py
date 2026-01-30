import json
import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.analysis import parameter_count

from utils.preprocessing import preprocessing

# from detectron2.utils.visualizer import Visualizer, ColorMode     # original import / code
from utils.custom_visualizer import Visualizer, ColorMode           # use custom threshold for keypoint visualization


def instances_to_coco_json(instances, image_id):
    
    # extract single model outputs and convert to list format
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    boxes = instances.pred_boxes.tensor.tolist()
    keypoints = instances.pred_keypoints.tolist()
    assert len(scores) == len(classes) == len(boxes) == len(keypoints)
    
    # gather full predictions as JSON serializable dictionary
    results = []
    for i in range(len(instances)):
        results.append({
            "image_id"      : image_id,
            "category_id"   : int(classes[i]),
            "score"         : float(scores[i]),
            "bbox"          : boxes[i],         # [xmin, ymin, xmax, ymax]
            "bbox_mode"     : "XYXY_ABS",
            "keypoints"     : keypoints[i],     # [x1, y1, v1, ..., xN, yN, vN] (v = visibility score)
        })

    return results


def get_args_parser():
    parser = argparse.ArgumentParser('Perform model inference', add_help=False)
    parser.add_argument('--download_dir', type=str, default=None,
                        help='Path to directory containing model checkpoints.')
    parser.add_argument('--model_type', type=str, default="Mask-R-CNN", choices=["Faster-R-CNN", "Mask-R-CNN"],
                        help='Model type used for performing inference.')
    return parser


if __name__ == '__main__':
    
    """
    This script performs object detection on TIFF projection images using a pre-trained Detectron2 model 
    (either Faster or Mask R-CNN) to identify implants commonly used in orthopedic and trauma procedures.
    It processes images from a specified directory, applies the model to perform inference, 
    and saves all predictions as PNG as well as JSON files.
    """

    # retrieve input arguments
    parser = argparse.ArgumentParser('Perform model inference', parents=[get_args_parser()])
    args = parser.parse_args()

    download_dir = Path(args.download_dir)
    model_type = args.model_type

    repo_dir = Path(__file__).resolve().parent
    cfg_file = repo_dir / "assets" / "config_files" / f"Implant_Detector_{model_type.replace('-', '_')}_Config.yaml"
    checkpoint = download_dir / f"Implant_Detector_{model_type.replace('-', '_')}.pth"

    # set input / output directory paths
    in_folder = repo_dir / "assets" / "input_images"
    eval_scenes = [d for d in in_folder.iterdir() if d.is_dir()]

    out_folder = repo_dir / "assets" / "model_predictions" / model_type.replace('-', '_')
    out_folder.mkdir(exist_ok=True, parents=False)

    # initialize meta information about the dataset
    MetadataCatalog.get("implant_detection").thing_classes = ["screw", "tulip_short", "tower", "tulip_long", "wire"]
    MetadataCatalog.get("implant_detection").thing_colors = [(255, 26, 26), (0, 204, 255), (153, 51, 255), (0, 255, 102), (255, 179, 0)]
    MetadataCatalog.get("implant_detection").keypoint_names = ["tip", "tip_direction", "head", "head_direction"]
    MetadataCatalog.get("implant_detection").keypoint_connection_rules = [("tip", "tip_direction", (2, 102, 236)),
                                                                          ("head", "head_direction", (13, 209, 99))]
    meta_data = MetadataCatalog.get("implant_detection")

    # setup model config file
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.WEIGHTS = str(checkpoint) # path to model checkpoint used for inference
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # configure predictor with a custom test time threshold

    # setup model predictor
    predictor = DefaultPredictor(cfg)

    param_dict = parameter_count(predictor.model)
    total_params = sum(param_dict.values())
    print(f"\nTotal model params: {total_params:,}")

    num_trainable = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)
    print(f"Trainable model params: {num_trainable:,}")

    # process each sample in the input folder
    print(f"\nFound {len(eval_scenes)} scenes for evaluation ...")
    for scene_dir in eval_scenes:
        scene_name = scene_dir.name
        print(f"\nPerforming inference for scene '{scene_name}' ...")
        
        # create output directories for this scene
        tmp_out_folder = out_folder / scene_name
        tmp_out_folder.mkdir(exist_ok=True, parents=False)
        tmp_out_folder_json = tmp_out_folder / "JSON-Files"
        tmp_out_folder_json.mkdir(exist_ok=True, parents=False)
        tmp_out_folder_png = tmp_out_folder / "PNG-Files"
        tmp_out_folder_png.mkdir(exist_ok=True, parents=False)

        eval_images = [f for f in scene_dir.iterdir() if f.is_file()]
        for eval_image in eval_images:
            view_name = eval_image.stem
            print(f"\t- Processing image '{view_name}'")

            # load and preprocess TIFF projection image
            img = tiff.imread(eval_image)
            assert img.dtype == np.float32
            img = preprocessing(img, clahe=False, subtract_lowpass=False, return_uint8=True)
            img = np.stack([img, img, img]).transpose((1, 2, 0)) # transpose to (H, W, C) format
            assert img.dtype == np.uint8

            # perform prediction on the image
            outputs = predictor(img)

            # visualize predictions
            v = Visualizer(img,
                           metadata=meta_data,
                           scale=1.0,
                           instance_mode=ColorMode.SEGMENTATION)
            
            instances = outputs["instances"].to("cpu")
            out = v.draw_instance_predictions(instances)

            # save predictions as JSON file            
            json_predictions = instances_to_coco_json(instances, view_name)
            pred_json_file = tmp_out_folder_json / f"{view_name}_pred.json"
            with open(pred_json_file, "w") as f:
                json.dump(json_predictions, f, indent=4)       

            # save visualized predictions as PNG image
            fig, ax = plt.subplots(figsize=(5, 5), frameon=False)
            ax.imshow(out.get_image(), cmap="gray")
            ax.axis('off')
            fig.tight_layout()
            fig.savefig(tmp_out_folder_png / f"{view_name}_pred.png",
                        bbox_inches='tight', pad_inches=0)
            # plt.show() # interactive mode only for debugging / testing
            plt.close()
