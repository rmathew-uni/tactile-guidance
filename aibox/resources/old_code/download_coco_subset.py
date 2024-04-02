import fiftyone.zoo as foz

dataset_dir = "/Users/florian/Documents/Studium/NBP/Projects/OptiVisT/AIBox/Dataset/Datasets/coco"

dataset = foz.load_zoo_dataset(
    "coco-2017",
    splits=["train", "validation", "test"],
    label_types=["detections"],
    classes=["banana"],
    max_samples=50,
    dataset_dir = dataset_dir
)

#session = fo.launch_app(dataset)
#session.dataset = dataset