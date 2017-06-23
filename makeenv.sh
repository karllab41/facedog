# Download conda and install it with:
curl https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash ~/Downloads/Miniconda3-latest-MacOSX-x86_64.sh 

# Create a conda environment and get into it
conda create -n face-recognition anaconda python=3.5
source activate face-recognition

# Install the necessary libraries
conda install -c menpo dlib
conda install -c https://conda.binstar.org/menpo opencv3
pip install scikit-image
pip install git+https://github.com/ageitgey/face_recognition_models

