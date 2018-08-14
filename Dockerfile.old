FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER Gabriel Garza <garzagabriel@gmail.com>

ARG TENSORFLOW_VERSION=1.6.0
ARG TENSORFLOW_ARCH=gpu
ARG KERAS_VERSION=2.1.5

#RUN echo -e "\n**********************\nNVIDIA Driver Version\n**********************\n" && \
#	cat /proc/driver/nvidia/version && \
#	echo -e "\n**********************\nCUDA Version\n**********************\n" && \
#	nvcc -V && \
#	echo -e "\n\nBuilding your Deep Learning Docker Image...\n"

# Install some dependencies
RUN apt-get update && apt-get install -y \
		bc \
		build-essential \
		cmake \
		curl \
		g++ \
		gfortran \
		git \
		libffi-dev \
		libfreetype6-dev \
		libhdf5-dev \
		libjpeg-dev \
		liblcms2-dev \
		libopenblas-dev \
		liblapack-dev \
		libopenjpeg2 \
		libpng12-dev \
		libssl-dev \
		libtiff5-dev \
		libwebp-dev \
		libzmq3-dev \
		nano \
		pkg-config \
		python-dev \
		software-properties-common \
		unzip \
		vim \
		wget \
		zlib1g-dev \
		qt5-default \
		libvtk6-dev \
		zlib1g-dev \
		libjpeg-dev \
		libwebp-dev \
		libpng-dev \
		libtiff5-dev \
		libjasper-dev \
		libopenexr-dev \
		libgdal-dev \
		libdc1394-22-dev \
		libavcodec-dev \
		libavformat-dev \
		libswscale-dev \
		libtheora-dev \
		libvorbis-dev \
		libxvidcore-dev \
		libx264-dev \
		yasm \
		libopencore-amrnb-dev \
		libopencore-amrwb-dev \
		libv4l-dev \
		libxine2-dev \
		libtbb-dev \
		libeigen3-dev \
		python-dev \
		python-tk \
		python-numpy \
		python3-dev \
		python3-tk \
		python3-numpy \
		ant \
		default-jdk \
		doxygen \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/* && \
# Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)
	update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3

# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
	rm get-pip.py

# Add SNI support to Python
RUN pip --no-cache-dir install \
		pyopenssl \
		ndg-httpsclient \
		pyasn1

# Install useful Python packages using apt-get to avoid version incompatibilities with Tensorflow binary
# especially numpy, scipy, skimage and sklearn (see https://github.com/tensorflow/tensorflow/issues/2034)
RUN apt-get update && apt-get install -y \
		python-numpy \
		python-scipy \
		python-nose \
		python-h5py \
		python-skimage \
		python-matplotlib \
		python-pandas \
		python-sklearn \
		python-sympy \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

# Install other useful Python packages using pip
RUN pip --no-cache-dir install numpy scipy sklearn scikit-image pandas matplotlib Cython requests pandas

# Install TensorFlow
RUN pip --no-cache-dir install \
	https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_ARCH}/tensorflow_${TENSORFLOW_ARCH}-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl

# Install Keras
RUN pip --no-cache-dir install git+git://github.com/fchollet/keras.git@${KERAS_VERSION}


# Install OpenCV
RUN git clone --depth 1 https://github.com/opencv/opencv.git /root/opencv && \
	cd /root/opencv && \
	mkdir build && \
	cd build && \
	cmake -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON -DWITH_XINE=ON -DBUILD_EXAMPLES=ON .. && \
	make -j"$(nproc)"  && \
	make install && \
	ldconfig && \
	echo 'ln /dev/null /dev/raw1394' >> ~/.bashrc

# PyCocoTools
#
# Using a fork of the original that has a fix for Python 3.
# I submitted a PR to the original repo (https://github.com/cocodataset/cocoapi/pull/50)
# but it doesn't seem to be active anymore.
RUN pip install --no-cache-dir git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI

# Expose Ports for TensorBoard (6006), Ipython (8888)
EXPOSE 6006 8888

WORKDIR "/root"
CMD ["/bin/bash"]

#
# # Essentials: developer tools, build tools, OpenBLAS
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     apt-utils git curl vim unzip openssh-client wget \
#     build-essential cmake \
#     libopenblas-dev
#
# #
# # Python 3.5
# #
# # For convenience, alias (but don't sym-link) python & pip to python3 & pip3 as recommended in:
# # http://askubuntu.com/questions/351318/changing-symlink-python-to-python3-causes-problems
# RUN apt-get install -y --no-install-recommends python3.5 python3.5-dev python3-pip python3-tk && \
#     pip3 install --no-cache-dir --upgrade pip setuptools && \
#     echo "alias python='python3'" >> /root/.bash_aliases && \
#     echo "alias pip='pip3'" >> /root/.bash_aliases
# # Pillow and it's dependencies
# RUN apt-get install -y --no-install-recommends libjpeg-dev zlib1g-dev && \
#     pip3 --no-cache-dir install Pillow
# # Science libraries and other common packages
# RUN pip3 --no-cache-dir install \
#     numpy scipy sklearn scikit-image pandas matplotlib Cython requests pandas
#
# #
# # Jupyter Notebook
# #
# # Allow access from outside the container, and skip trying to open a browser.
# # NOTE: disable authentication token for convenience. DON'T DO THIS ON A PUBLIC SERVER.
# RUN pip3 --no-cache-dir install jupyter && \
#     mkdir /root/.jupyter && \
#     echo "c.NotebookApp.ip = '*'" \
#          "\nc.NotebookApp.open_browser = False" \
#          "\nc.NotebookApp.token = ''" \
#          > /root/.jupyter/jupyter_notebook_config.py
# EXPOSE 8888
#
# #
# # Tensorflow 1.6.0 - GPU
# #
# # Install TensorFlow
# RUN pip --no-cache-dir install \
# 	https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_ARCH}/tensorflow_${TENSORFLOW_ARCH}-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl
#
# # Expose port for TensorBoard
# EXPOSE 6006
#
# #
# # OpenCV 3.4.1
# #
# # Dependencies
# RUN apt-get install -y --no-install-recommends \
#     libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
#     libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libgtk2.0-dev \
#     liblapacke-dev checkinstall
# # Get source from github
# RUN git clone -b 3.4.1 --depth 1 https://github.com/opencv/opencv.git /usr/local/src/opencv
# # Compile
# RUN cd /usr/local/src/opencv && mkdir build && cd build && \
#     cmake -D CMAKE_INSTALL_PREFIX=/usr/local \
#           -D BUILD_TESTS=OFF \
#           -D BUILD_PERF_TESTS=OFF \
#           -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
#           .. && \
#     make -j"$(nproc)" && \
#     make install
#
# #
# # Keras 2.1.5
# #
# RUN pip3 install --no-cache-dir --upgrade h5py pydot_ng keras
#
# #
# # PyTorch 0.3.1
# #
# RUN pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp35-cp35m-linux_x86_64.whl && \
#     pip3 install torchvision
#
# #
# # PyCocoTools
# #
# # Using a fork of the original that has a fix for Python 3.
# # I submitted a PR to the original repo (https://github.com/cocodataset/cocoapi/pull/50)
# # but it doesn't seem to be active anymore.
# RUN pip3 install --no-cache-dir git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI
#
# WORKDIR "/root"
# CMD ["/bin/bash"]
