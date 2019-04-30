import colorsys
import random
import numpy as np
from skimage.measure import label, regionprops, find_contours
from matplotlib.patches import Polygon
from matplotlib import patches,  lines
import matplotlib.pyplot as plt
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None, filename=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        fig, ax = plt.subplots(1, figsize=figsize)
        auto_show = True
    fig.tight_layout()

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        x1, y1, x2, y2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='green', size=15, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if filename : 
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
    else :
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data 


def rleToMask(rleString,height,width):
  rows,cols = height,width
  rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
  rlePairs = np.array(rleNumbers).reshape(-1,2)
  img = np.zeros(rows*cols,dtype=np.uint8)
  for index,length in rlePairs:
    index -= 1
    img[index:index+length] = 1
  img = img.reshape(cols,rows)
  img = img.T
  return img

def makeBBox(mask) :
    lbl_0 = label(mask)
    props = regionprops(lbl_0)
    height, width = mask.shape
    y1, x1, y2, x2 = height, width, 0, 0
    for prop in props :
        minr, minc, maxr, maxc = prop.bbox
        y1 = min(y1, minr)
        x1 = min(x1, minc)
        y2 = max(y2, maxr)
        x2 = max(x2, maxc)
    bbox = np.array([y1, x1, y2, x2])
    return bbox

def viz_coco_example(img, imginfo, annotations, classnames) :
    height, width = imginfo['height'], imginfo['width']
    assert(img.shape[0] == height)
    assert(img.shape[1] == width)
    masks = SegmentationMask([ann['segmentation'] for ann in annotations], size=(width, height), mode='poly')
    bboxes = []
    classids = []
    for ann in annotations :
        x, y, width, height = ann['bbox']
        x1, y1, x2, y2 = x, y, x+width, y+height
        bboxes.append([y1, x1, y2, x2])
        classids.append(ann['category_id']-1) # category_ids are 1-indexed in COCO datasets
    classids = np.array(list(map(int, classids)))
    bboxes = np.stack(bboxes, axis=0)
    masks = masks.get_mask_tensor().numpy()
    if masks.ndim == 2 :
        masks = masks[np.newaxis, :, :]
    masks = np.transpose(masks, (1, 2, 0))
    display_instances(img, boxes=bboxes, masks=masks, class_ids=classids, class_names=classnames)
