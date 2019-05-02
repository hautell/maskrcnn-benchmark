import os 
import torch 
import json
import random
from PIL import Image

class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(
        self, ann_file, root, transforms=None
    ):
        with open(ann_file, 'r') as f :
            self.anns = json.load(f)
        self.anns = [(pn,category) for pn,category in self.anns.items()]
        self.anns = self.anns
        random.shuffle(self.anns)
        self.root = root
        self.transforms = transforms

        self.categories = [
            "__background",
            "bag",
            "belt",
            "outer",
            "dress",
            "sunglasses",
            "pants",
            "top",
            "shorts",
            "skirt",
            "headwear",
            "scarf & tie"
        ]


    def __getitem__(self, idx):

        product_name = self.anns[idx][0]
        try :
            img1 = Image.open(os.path.join(self.root, product_name + '-0' + '.jpg')).convert('RGB')
            img2 = Image.open(os.path.join(self.root, product_name + '-1' + '.jpg')).convert('RGB')
        except :
            return self.__getitem__(idx+1)

        img1 = self.transforms(img1)
        img2 = self.transforms(img2)

        category = self.anns[idx][1]
        category_id = self.categories.index(category)
        return img1, img2, category_id

        # filter crowd annotations
        # TODO might be better to add an extra field
        # anno = [obj for obj in anno if obj["iscrowd"] == 0]

        # boxes = [obj["bbox"] for obj in anno]
        # boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        # target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        # classes = [obj["category_id"] for obj in anno]
        # classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        # classes = torch.tensor(classes)
        # target.add_field("labels", classes)

        # masks = [obj["segmentation"] for obj in anno]
        # masks = SegmentationMask(masks, img.size, mode='poly')
        # target.add_field("masks", masks)

        # if anno and "keypoints" in anno[0]:
        #     keypoints = [obj["keypoints"] for obj in anno]
        #     keypoints = PersonKeypoints(keypoints, img.size)
        #     target.add_field("keypoints", keypoints)

        # target = target.clip_to_image(remove_empty=True)

        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

        # return img, target, idx

    def __len__(self) :
        return len(self.anns)
