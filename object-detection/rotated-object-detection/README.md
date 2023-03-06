# Rotated Object Detection

Rotated Object Detection is a computer vision technique used to detect and locate objects that are not aligned with the horizontal and vertical axes. This is particularly useful in scenarios where objects are captured in images or videos from varying angles or orientations.

The task of detecting rotated objects can be challenging because traditional object detection methods assume that objects are axis-aligned. Therefore, rotated object detection requires more advanced techniques that take into account the orientation and viewpoint of the object.

## Common approaches:

- **Rotation-invariant methods:** These methods transform the input image to align the object with the horizontal and vertical axes before using traditional object detection algorithms.
- **Anchor-free methods:** These methods use anchor-free techniques to detect objects without assuming a fixed orientation. Instead, they detect the object by identifying key points on the object and estimating its rotation angle.
- **Rotation-sensitive methods:** These methods use region proposals to detect objects and then predict the orientation of each object using regression techniques.

## Dataset:

Some popular datasets used in rotated object detection research include 

- DOTA
- HRSC2016
- UCAS-AOD

These datasets provide annotated images of objects with varying orientations and viewpoints.

## Research

Research in rotated object detection is ongoing, and there are several promising papers in this field. Here are a few notable papers:

- [Rotation-aware Object Detection in Aerial Images, by Y. Wu et al. (2017)](https://arxiv.org/abs/1711.06728)
- [Arbitrary-Oriented Object Detection with Circular Smooth Label, by Y. Liu et al. (2019)](https://arxiv.org/abs/1908.03673)
- [Deep Rotated Bounding Box Encoding for Object Detection, by H. Dai et al. (2019)](https://arxiv.org/abs/1908.05612)
- [Arbitrary-Oriented Scene Text Detection via Rotation Proposals, by M. Liao et al. (2018)](https://arxiv.org/abs/1711.0908)
- [Shape Robust Text Detection with Progressive Scale Expansion Network, by X. Li et al. (2019)](https://arxiv.org/abs/1903.12473)

By exploring these papers and datasets, researchers and developers can stay up-to-date on the latest techniques and advancements in rotated object detection.
