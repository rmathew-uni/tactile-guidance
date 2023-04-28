import fiftyone as fo
import fiftyone.zoo as foz

dataset_dir = "C:/Users/powel/fiftyone/coco-2017/banana2" \
              ""
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=["banana"],
    max_samples=100,
    dataset_dir = dataset_dir
)

session = fo.launch_app(dataset)

session.dataset = dataset
