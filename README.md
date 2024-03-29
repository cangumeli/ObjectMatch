# ObjectMatch: ObjectMatch: Robust Registration using Canonical Object Correspondences (CVPR 2023)
Offical code for our CVPR '23 paper ObjectMatch: Robust Registration using Canonical Object Correspondences. Project website, arxiv paper and video through [this link](https://cangumeli.github.io/ObjectMatch/). This repo is work in progress, stay tuned for training/evaluation scripts, training data, and additional experiments.

![](https://cangumeli.github.io/ObjectMatch/static/images/teaser.jpg)

## Installation
This code is mainly developed under Ubuntu 20.04 with an RTX 3090 GPU (w/ CUDA 11) and an Intel Xeon CPU. We use anaconda3 with Python 3.9 as the base Python setup.

After cloning this repo, please run:
```
source setup.sh
```
Now, you should have a conda environment names `objectmatch` with all the required libraries. You can activate it by running
```
conda activate objectmatch
```

Running the `setup.sh` will also download our [SuperGlue fork](https://github.com/cangumeli/SuperGluePretrainedNetwork) (Sarlin et al. '20) to the parent directory of the code. SuperGlue complements our object correspondences by providing off-the-shelf keypoint matches.

## Downloading Pre-trained Models and ScanNet Test Images
We share pre-trained networks and pre-processed test images [here](https://drive.google.com/drive/folders/1ATd3A7_ry4_6NbRDLfywKvtntsZS_UZE?usp=drive_link). Both archives are very small with sizes around ~1GB.

`checkpoints.zip` archive contains the weights of NOC prediction and object identification networks. Unzip it to the root directory of the code.

`TestImages.zip` contains a set of test and validation images from ScanNet (Dai et al. '17). We use these images for demos and evaluation. Download and unzip it to the same directory as the code for your convenience.

Alternatively, you can test the model over the sample images provided at `assets/sample_pairs`. You may also use the directory structure there as a format for your own evaluations.

## Running Code
Check out `pair_eval.py` for running pairwise registration. To run and visualize some demo samples we provided, simply run:
```
python pair_eval.py --file ./assets/samples.txt --test_folder ./assets/sample_pairs --vis
```

You should see visualizations in the `vis_pairs` folder. Each subfolder contains input frames, NOCs and keypoint matches (where applicable), and a set of meshes that show resulting registrations.

## Citation
If you find our work or this repo useful, please cite:
```
@inproceedings{gumeli2023objectmatch,
  title={ObjectMatch: Robust Registration using Canonical Object Correspondences},
  author={G{\"u}meli, Can and Dai, Angela and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13082--13091},
  year={2023}
}
```
