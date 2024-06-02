# ALPS
This is the offical implement of our ALPS framwork.

Paper: "ALPS: An Auto-Labeling and Pre-training Scheme for Remote Sensing Segmentation With Segment Anything Model"

## Usage
### Environment
- Python 3.8.5
- Pytorch 1.12.1+cu113
- Torchvision 0.13.1+cu113
- Torchaudio 0.12.1+cu113
- scikit-learn 1.3.2
- segment-anything 1.0

Install Segment Anything:
```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

Click the links below to download the checkpoint for the corresponding model type.
- vit_h: [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- vit_l: [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- vit_b: [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)



### Getting Started
1. Follow the structure below to organize the dataset:
```
├──dataset_path
│   ├── img_dir
│   │   ├── train
│   │   │   ├── xxx{img_suffix}
│   │   │   ├── yyy{img_suffix}
│   │   │   ├── zzz{img_suffix}
│   │   │   ├── ....
│   │   ├── val
│   │   │   ├── xxx{img_suffix}
│   │   │   ├── yyy{img_suffix}
│   │   │   ├── zzz{img_suffix}
│   │   │   ├── ....
│   │   ├── test
│   │   │   ├── xxx{img_suffix}
│   │   │   ├── yyy{img_suffix}
│   │   │   ├── zzz{img_suffix}
│   │   │   ├── ....
```
2. First, download the pre-trained SAM model checkpoint. Then, you can use the command line below to perform automatic labeling on your own datasets:
```
python main.py --root_dir .../dataset_path --image_suffix .png --sam_checkpoint .../sam_vit_h_4b8939.pth --model_type vit_h --number_clusters xx --vis True
```

3. Additionally, our framework supports more customized parameter settings:
- 'process_list': This can be set to ‘train’, ‘val’, or ‘test’.
- 'threshold': This is the area size threshold used in the filtering gate module.
- 'label_dir': This is the path where the pseudo-labels are saved.
- 'device': This can be set to ‘cuda’ or ‘cpu’.
- 'batch_size': This is the batch size used in online K-means.
- More parameter settings can be found in the main.py file.