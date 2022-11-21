import os

from functions import train

os.system("git clone https://github.com/WongKinYiu/yolov7.git")
os.system('wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt')

from simple_image_download import simple_image_download as smp
response = smp.simple_image_download
keys = ["Jack Sparrow"]
for k in keys:
    response().download(k, 150)

train(
    dataset_path='simple_images',
    batch=4,
    epochs=50,
    name='jack_sparrow',
    save_path='simple_images'
)
