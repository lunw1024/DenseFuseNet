<img src="https://github.com/IDl0T/DenseFuseNet/blob/master/Yau-Photo.jpg" width = "300"/> <img src="https://github.com/IDl0T/DenseFuseNet/blob/master/Yau-Award.jpg" width = "300"/> <img src="https://github.com/IDl0T/DenseFuseNet/blob/master/Yau-Trophy.jpg" width = "300"/>

[Paper Accepted to ICCECE 2021](https://github.com/IDl0T/DenseFuseNet/blob/master/Notification%20of%20Paper%20Acceptance.pdf)

# [DenseFuseNet: Improve 3D Semantic Segmentation in the Context of Autonomous Driving with Dense Correspondence](https://github.com/IDl0T/DenseFuseNet/blob/master/DenseFuseNet.pdf)
*Yulun Wu*
## Abstract
With the development of deep convolutional networks, autonomous driving has been reforming human social activities in the recent decade. The core issue of autonomous driving is how to integrate the multi-modal perception system effectively, that is, using sensors such as lidar, RGB camera, and radar to identify general objects in traffic scenes. Extensive investigation shows that lidar and cameras are the two most powerful sensors widely used by autonomous driving companies such as Tesla and Waymo, which indeed revealed that how to integrate them effectively is bound to be one of the core issues in the field of autonomous driving in the future. Obviously, these two kinds of sensors have their inherent advantages and disadvantages. Based on the previous research works, we are motivated to fuse lidars and RGB cameras together to build a more robust perception system.

It is not easy to design a model with two different domains from scratch, and a large number of previous works (e.g., FuseSeg) has sufficiently proved that merging the RGB camera and lidar models can attain better results on vision tasks than the lidar model alone. However, it cannot adequately handle the inherent correspondence between the RGB camera and lidar data but rather arbitrarily interpolates between them, which quickly leads to severe distortion, heavy computational burden, and diminishing performance.

To address these problems, in this paper, we proposed a general framework to establish a connection between lidar and RGB camera sensors, matching and fusing the features of the lidar and RGB models. We also defined two kinds of inaccuracies (missing pixels and covered points) in spherical projection and conducted a numerical analysis on them. Furthermore, we proposed an efficient filling algorithm to remedy the impact of missing pixels. Finally, we proposed a 3D semantic segmentation model, DenseFuseNet, which incorporated our techniques and achieved a noticeable 5.8 and 14.2 improvement in mIoU and accuracy on top of vanilla SqueezeSeg. All code is already open-source on https://github.com/IDl0T/DenseFuseNet.

***Keywords**: Autonomous driving, 3D semantic segmentation, lidar, point clouds, sensor fusion, spher-
ical projection, noise analysis, convolutional neural networks*
