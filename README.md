# Real time natural disaster assessment using AI

Natural disasters provide many challenges to governments and rescue services. We propose AI based tool that help to provide humanitarian assistance & disaster recovery. Currently this is done manually or with an AI to a very limited extend.

## The approach
### Background
To finetune for these tasks in this repository, we make use of [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/), which provides an extensible framework for segmentation tasks. 

[MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) allows us to concatenate necks and heads appropriate for any segmentation downstream task to the encoder, and then perform the finetuning. This only requires setting up a config file detailing the desired model architecture, dataset setup and training strategy. 

### The pretrained backbone
The pretrained model we work with is a [ViT](https://arxiv.org/abs/2010.11929) operating as a [Masked Autoencoder](https://arxiv.org/abs/2111.06377), trained on [HLS](https://hls.gsfc.nasa.gov/) data. The encoder from this model is made available as the backbone and the weights can be downloaded from Hugging Face [here](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/blob/main/Prithvi_100M.pt).

### The architectures
We use a simple architecture that adds a neck and segmentation head to the backbone. The neck concatenates and processes the transformer's token based embeddings into an embedding that can be fed into convolutional layers. The head processes this embedding into a segmentation mask. The code for the architecture can be found in [this file](./ML Model/geospatial_fm/geospatial_fm.py).

## Setup
### Dependencies
1. Clone this repository
2. `conda create -n <environment-name> python==3.9`
3. `conda activate <environment-name>`
4. Install torch (tested for >=1.7.1 and <=1.11.0) and torchvision (tested for >=0.8.2 and <=0.12). May vary with your system. Please check at: https://pytorch.org/get-started/previous-versions/.
    1. e.g.: `pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 --extra-index-url https://download.pytorch.org/whl/cu115`
5. `cd` into the cloned repo
5. `pip install -e .`
6. `pip install -U openmim`
7. `mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/{cuda_version}/{torch_version}/index.html`. Note that pre-built wheels (fast installs without needing to build) only exist for some versions of torch and CUDA. Check compatibilities here: https://mmcv.readthedocs.io/en/v1.6.2/get_started/installation.html
    1. e.g.: `mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11.0/index.html`

### Data

The xBD dataset can be downloaded from [xView2 Competition](https://xview2.org/).

## Running the finetuning

1. With the conda env created above activated, run:

`mim train mmsegmentation --launcher pytorch configs/xView2.py`

2. To run testing: 

`mim test mmsegmentation configs/xView2.py --checkpoint /path/to/best/checkpoint/model.pth --eval "mIoU"`


## Running the inference
We provide a script to run inference on new data in GeoTIFF format. The data can be of any shape (e.g. height and width) as long as it follows the bands/channels of the original dataset. An example is shown below.

```
python model_inference.py -config /path/to/config/config.py -ckpt /path/to/checkpoint/checkpoint.pth -input /input/folder/ -output /output/folder/ -input_type tif -bands "[0,1,2,3,4,5]"
```

The `bands` parameter is useful in case the files used to run inference have the data in different orders/indexes than the original dataset.
