# Black_white
Black & White Image Colorization 🎨🖤🤍

This project uses a deep learning model to colorize black and white images using OpenCV's DNN module and a pre-trained Caffe model.

Requirements
------------
- Python 3.x
- OpenCV (cv2)
- NumPy

You can install the dependencies using:

    pip install numpy opencv-python

Files
-----
- colour.py: Colorizes a single grayscale image (image.jpg) and displays the result side by side.
- run.py: Colorizes all images in a given directory (images/) in batch mode.
- colorization_deploy_v2.prototxt: Model architecture definition.
- colorization_release_v2.caffemodel: Pre-trained model weights (not included here).
- pts_in_hull.npy: Cluster centers in the ab color space.

Sample Images
-------------
You can use your own grayscale images, or use the provided ones:
- image.jpg, image3.jpg, image4.jpg, images1.jpg, Landscape-BW.jpg

How to Run
----------
Single Image Colorization:

    python colour.py

This will read image.jpg, colorize it, and display the grayscale and colorized versions side by side.

Batch Colorization:

    python run.py

This will process all images in the images/ directory and display them one by one for 2 seconds each.

Directory Structure
-------------------
project_root/
│
├── colour.py
├── run.py
├── colorization_deploy_v2.prototxt
├── colorization_release_v2.caffemodel
├── pts_in_hull.npy
├── image.jpg
├── images/         # Folder with images for batch processing
│   ├── image1.jpg
│   ├── ...

Credits
-------
- Based on OpenCV's colorization model.
- Model trained by Richard Zhang et al. (ECCV 2016): https://richzhang.github.io/colorization/
