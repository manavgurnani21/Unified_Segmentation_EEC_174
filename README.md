# EcoCar Object and Lane Segmentation
## EEC174 Final Project
By Rami Abudamous, Gregory Ceron, Siyun Chen, Tyson-Tien Nguyen, and Manav Gurnani

### Installation:
**Requirements:**
- The model was run on Windows 11 using a WSL Ubuntu environment
- An NVIDIA GPU with:
 - nvdec/nvenc support
 - CUDA version 11.x or newer
- FFmpeg with nvenc support

**Installation Steps**:
- Install the requirements from `requirements.txt`
- Ensure that NVIDIA DALI is matched to the version of CUDA (ex cuda120 for 12.0, cuda118 for 11.8, etc). It may need to be installed from the NVIDIA index:
```
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda120
```
Note that this installs the version for cuda120

### Usage:
**Preparation:**
- create a `/weights/` directory and download the [weights](https://drive.google.com/file/d/1dlwaElu0dQQdoEeJkuP2LKGx1TSCjE-z/view)
- create a `/inputs/` directory and move image or video files there

**Inference:**
Run the following command. The output will be in `/inference/output/` by default. This can be changed in the command:
```
python3 tools/inference.py --weights weights/epoch-195.pth --source inputs/{video or image name} --save-dir inference/output --batch-size 16 --device 0 
```