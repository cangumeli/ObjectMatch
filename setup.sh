# create env
conda create -n objectmatch python=3.9
conda activate objectmatch

# install pytorch
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia

# install detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# install 
conda install -c bottler nvidiacub
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d


pip install -r requirements.txt


cd ..
git clone https://github.com/cangumeli/SuperGluePretrainedNetwork.git
cd ObjectMatch
