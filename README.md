# Summary 

A Python Wrapper around the Optical Flow method developed during Cei Lui's PhD Thesis. Please visit his webpage ([linked here](https://people.csail.mit.edu/celiu/OpticalFlow/)) for more information. As of Sep 2nd, 2021 this method is ranked 197 on the KITTI 2015 leaderboard for Optical Flow.

# Deps

The wrapper was made on a system running Ubuntu 20 LTS. The project depends on SWIG and OpenCV. To install SWIG:

```
sudo apt-get install -y swig
```

OpenCV installs are covered well online.


# Install

To install, please use the following instructions from the root directory

```
mkdir ./build
cd build
cmake ..
make
python ./pytests/test_optical_flow.py
```

# References

[1] S. Baker, D. Scharstein, J. Lewis, S. Roth, M. J. Black, and R. Szeliski. A database and evaluation methodology for optical flow. In Proc. IEEE International Conference on Computer Vision (ICCV), 2007.

[2] T. Brox, A. Bruhn, N. Papenberg, and J.Weickert. High accuracy optical flow estimation based on a theory for warping. In European Conference on Computer Vision (ECCV), pages 25–36, 2004.

[3] A. Bruhn, J.Weickert and C. Schn¨orr. Lucas/Kanade meets Horn/Schunk: combining local and global optical flow methods. International Journal of Computer Vision (IJCV), 61(3):211–231, 2005.

[4] C. Liu, W. T. Freeman, E. H. Adelson and Y. Weiss. Human-assisted motion annotation. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1-8, 2008.

[5] C. Liu. Beyond Pixels: Exploring New Representations and Applications for Motion Analysis. Doctoral Thesis. Massachusetts Institute of Technology. May 2009.

