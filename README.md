# semi_supervised_3d_semseg_AD

In this repository, we will work on creating a semi-supervised 3D semantic segmentation model for autonomous driving. The model will be trained on partial labels from the nuScenes dataset and pseudo-labels generated from projection of open vocabulary 2D semantic segmentation models of cameras to the LiDAR point cloud.

## Setup
1. Clone the repo
2. Go to the repo directory
```bash
cd semi_supervised_3d_semseg_AD
```
3. Create a docker image with the following command:
```bash
docker build -t 3d-semisuper-semseg .
```
4. Run the docker container with the following command:
```bash
docker run --name 3d_semisuper_semseg -it --gpus all -e DISPLAY=$DISPLAY -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR -v /path/to/datastore:/path/to/datastore -v `pwd`:/workspace -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg --shm-size=16g --network fiftyone_network -e FIFTYONE_DATABASE_URI=mongodb://fiftyone_server:27017 3d-semisuper-semseg 
```

You only add the ``-e DISPLAY=$DISPLAY -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg`` if youw want support for OpenGL visualization in the container and using WSLg for WSL2.

### Open3D setup

Install Open3D with the following steps:

1. clone the Open3D repository
```bash
git clone https://github.com/isl-org/Open3D
```

2. Install Open3D-ML within the Open3D repository
```bash
cd Open3D
git clone https://github.com/isl-org/Open3D-ML.git
```
3. create a build directory
```bash
mkdir build
cd build
```
4. build Open3D with Open3D-ML support. You can change the CUDA architecture to match your GPU. For example, for a RTX 3090, you can use 86, for V100, you can use 70, etc. 
```bash
cmake -DBUILD_CUDA_MODULE=ON \
      -DGLIBCXX_USE_CXX11_ABI=OFF \
      -DBUILD_PYTORCH_OPS=ON \
      -DBUILD_TENSORFLOW_OPS=OFF \
      -DBUNDLE_OPEN3D_ML=ON \
      -DOPEN3D_ML_ROOT=/workspace/Open3D/Open3D-ML \
      -DCMAKE_CUDA_ARCHITECTURES="75" \
      ..
```

5. Make Open3D for python. You can change the number of cores to use with the -j flag. example: -j4 for 4 cores.
```bash
make -j4 install-pip-package
```

