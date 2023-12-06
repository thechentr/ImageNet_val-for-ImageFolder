# ImageNet_val

该数据集通过以命令下载

```bash
git clone https://github.com/thechentr/ImageNet_val-for-ImageFolder.git
cd ImageNet_val-for-ImageFolder
mkdir ILSVRC2012_img_val
wget wget -P ILSVRC2012_img_val https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
tar -xf ILSVRC2012_img_val.tar -C ILSVRC2012_img_val
```

并通过脚本整理获得

```bash
python convert.py
```



## 使用方法

只需要使用`torchvision.datasets.ImageFolder`即可：

```python
# 加载 ImageNet 测试集
from torchvision import datasets, transforms
imagenet_val = datasets.ImageFolder('ILSVRC2012_img_val_for_ImageFolder', transform=transform)
imagenet_iter = DataLoader(imagenet_val, batch_size=64, shuffle=False)
```

详细见`accuracy.py`

