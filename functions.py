import os
import shutil


def predict(weights, conf, source, save_path, name):
    try:
        shutil.rmtree(f"{save_path}/{name}")
    except:
        pass
    os.system(
        f"python yolov7/detect.py --weights {weights} --conf {conf} "
        f"--img-size 640 --source {source} --save-txt --project {save_path} --name {name}"
    )


def train(dataset_path, batch, epochs, weights='yolo7/yolo7.pt'):
    try:
        os.remove(f"{dataset_path}/train/labels.cache")
        os.remove(f"{dataset_path}/val/labels.cache")
    except:
        pass

    os.system(
        f"python yolov7/train.py --workers 1 --device 0 --batch-size {batch} --epochs {epochs} --weights {weights} "
        f"--data {dataset_path}/data_custom.yaml --img-size 640 640 --cfg {dataset_path}/cfg_custom.yaml "
        f"--hyp {dataset_path}/hyp.scratch.custom.yaml"
    )
