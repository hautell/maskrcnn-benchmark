from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.modanetDrawer import ModaNetDrawer
import argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
parser.add_argument(
    "--config-file",
    default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
    metavar="FILE",
    help="path to config file",
)
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)

args = parser.parse_args()

# update the config options with the config file
cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
# manual override some options

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
image = np.array(Image.open('/data/looks/resort-2019_Gucci_095.jpg'))
result, top_prediction = coco_demo.run_on_opencv_image(image)
from IPython import embed; embed()
