# Contrastive Instance Association for 4D Panoptic Segmentation using Sequences of 3D LiDAR Scans

**Abstract -** 
  Scene understanding is critical for autonomous navigation in dynamic
  environments. Perception tasks in this domain like segmentation and tracking are usually
  tackled individually.
  In this paper, we address the problem of 4D panoptic segmentation using LiDAR scans, which
  requires to assign to each 3D point in a temporal sequence of scans a semantic
  class and for each object a temporally consistent instance ID.
  We propose a novel approach that builds on top of an arbitrary single-scan panoptic segmentation
  network and extends it to the temporal domain by associating instances across time.
  We propose a contrastive aggregation network that leverages the point-wise features from the panoptic
  network. It generates an embedding space in which encodings of the same instance at different
  timesteps lie close together and far from encodings belonging to other instances. The training
  is inspired by contrastive learning techniques for self-supervised metric learning. Our
  association module combines appearance and motion cues to associate instances across 
  scans, allowing us to perform temporal perception. We evaluate our
  proposed method on the SemanticKITTI benchmark and achieve state-of-the-art results even
  without relying on pose information.


Source code for our work soon to be published at RA-L:

```
@article{marcuzzi2022ral,
  author = {Rodrigo Marcuzzi and Lucas Nunes and Louis Wiesmann and Ignacio Vizzo and Jens Behley and Cyrill Stachniss},
  title = {{Contrastive Instance Association for 4D Panoptic Segmentation \\ using Sequences of 3D LiDAR Scans}},
  journal = {IEEE Robotics and Automation Letters (RA-L)},
  year = 2022
}
```

More information on the article and code will be published soon. Stay tuned.
