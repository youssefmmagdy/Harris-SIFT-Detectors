# Computer Vision: Feature Detection & Robustness Analysis

A comprehensive analysis of corner detection and feature detection algorithms in computer vision, comparing **Harris Corner Detector** and **SIFT (Scale-Invariant Feature Transform)** techniques.

## 📋 Project Overview

This project explores two fundamental techniques in computer vision for detecting and matching features in images:

1. **Harris Corner Detector** - A traditional corner detection algorithm
2. **SIFT (Scale-Invariant Feature Transform)** - A modern, robust feature detection algorithm

The project includes implementation, performance evaluation, and robustness analysis under various image transformations.

## 🎯 Project Goals

- Implement and understand the Harris Corner Detection algorithm from first principles
- Learn about SIFT keypoint detection and feature descriptors
- Compare the performance of Harris vs SIFT on real-world images
- Evaluate robustness under image transformations (scaling, rotation, illumination changes, blur, noise)
- Provide visual demonstrations and quantitative metrics for algorithm comparison

## 🛠️ Technologies Used

- **Python 3.x**
- **OpenCV** (cv2) - Computer vision library
- **NumPy** - Numerical computing
- **Pandas** - Data analysis
- **Matplotlib** - Data visualization
- **Jupyter Notebook** - Interactive development environment

## 📊 Dataset

The project uses the [Google Image Recognition Tutorial](https://www.kaggle.com/datasets/wesamelshamy/google-image-recognition-tutorial) dataset from Kaggle, containing various images suitable for feature detection analysis.

<img width="1490" height="459" alt="1" src="https://github.com/user-attachments/assets/73b0adca-783c-4aac-8a28-aae13e04b685" />


## 📁 Project Structure

```
CV_Project/
├── CV_Project.ipynb          # Main Jupyter notebook with all implementations
├── report.pdf                # Detailed project report
├── README.md                 # This file
└── .git/                     # Version control
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install opencv-contrib-python
pip install numpy
pip install pandas
pip install matplotlib
pip install kagglehub
pip install jupyter
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CV_Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:
```bash
jupyter notebook CV_Project.ipynb
```

## 📝 Key Features

### 1. Harris Corner Detector

**Implementation Details:**
- Computes horizontal and vertical image derivatives using Sobel kernels
- Applies Gaussian smoothing to derivative products
- Computes corner response: R = (A×B - C²) - k(A + B)²
- Thresholds response to identify strong corners

**Key Parameters:**
- `k = 0.05` (Harris parameter)
- `threshold = 0.5` (corner response threshold)

#### Harris Corner Response Maps
<img width="1005" height="739" alt="2" src="https://github.com/user-attachments/assets/970afdd9-e42f-4893-9d6e-463f55939845" />


#### Thresholded Responses
<img width="1926" height="788" alt="3" src="https://github.com/user-attachments/assets/2afa86a2-59e0-4429-91ed-da1a67f0c959" />

<img width="1926" height="788" alt="4" src="https://github.com/user-attachments/assets/79903a98-22cc-4358-8b0e-7cc0c7d82268" />

**Strengths:**
- Fast computation
- Good for detecting sharp corners
- Simple implementation

**Limitations:**
- Not scale-invariant
- Not rotation-invariant
- Sensitive to image transformations

### 2. SIFT (Scale-Invariant Feature Transform)

**Implementation Details:**
- Detects keypoints across multiple scales using Difference of Gaussians (DoG)
- Computes rotation-invariant descriptors for each keypoint
- Uses Brute Force Matcher (BFMatcher) with L2 norm for feature matching

#### Gradient Analysis

![Gradient Analysis - Image 1](docs/images/figure_05.png)

![Gradient Analysis - Image 2](docs/images/figure_06.png)

![Gradient Analysis - Image 3](docs/images/figure_07.png)

![Gradient Analysis - Image 4](docs/images/figure_08.png)

![Gradient Analysis - Image 5](docs/images/figure_09.png)

#### SIFT Keypoint Detection

![SIFT Keypoints - Building 1](docs/images/figure_10.png)

![SIFT Keypoints - Building 2](docs/images/figure_11.png)

#### SIFT Feature Matching

![SIFT Feature Matches between Building 1 and Building 2](docs/images/figure_12.png)

**Strengths:**
- Scale-invariant
- Rotation-invariant
- Illumination-invariant
- Robust to image noise
- Excellent for image matching and recognition

**Limitations:**
- Computationally more expensive than Harris
- More complex implementation
- Sensitive to extreme transformations

## 📈 Analysis & Results

### Keypoint Detection Comparison

The project generates bar charts comparing the number of keypoints detected by Harris vs SIFT across the dataset images.

![Harris vs SIFT Keypoint Comparison](docs/images/figure_13.png)

### Robustness Analysis

The project evaluates both detectors under various transformations:

#### Image Transformations Applied:
- **Scaling**: 50%, 150%
- **Rotation**: 30°, 90°
- **Illumination Changes**: Brightness and contrast adjustments
- **Gaussian Blur**: Various kernel sizes
- **Gaussian Noise**: Different noise levels

#### Harris Corner Detector - Transformation Results:

**Original Image**
![Harris - Original (494112 keypoints)](docs/images/figure_14.png)

**Scaled 50%**
![Harris - Scaled 50% (138730 keypoints)](docs/images/figure_15.png)

**Scaled 150%**
![Harris - Scaled 150% (955133 keypoints)](docs/images/figure_16.png)

**Rotated 30°**
![Harris - Rotated 30° (452630 keypoints)](docs/images/figure_17.png)

**Rotated 90°**
![Harris - Rotated 90° (352827 keypoints)](docs/images/figure_18.png)

**Brighter (alpha=1.5, beta=30)**
![Harris - Brighter (296138 keypoints)](docs/images/figure_19.png)

**Darker (alpha=0.7, beta=-20)**
![Harris - Darker (584559 keypoints)](docs/images/figure_20.png)

**Blurred (kernel=7x7)**
![Harris - Blurred (362197 keypoints)](docs/images/figure_21.png)

**Noisy (std_dev=30)**
![Harris - Noisy (496073 keypoints)](docs/images/figure_22.png)

#### SIFT - Transformation Results:

**Original Image**
![SIFT - Original (6624 keypoints)](docs/images/figure_23.png)

**Scaled 50%**
![SIFT - Scaled 50% (2333 keypoints)](docs/images/figure_24.png)

**Scaled 150%**
![SIFT - Scaled 150% (9952 keypoints)](docs/images/figure_25.png)

**Rotated 30°**
![SIFT - Rotated 30° (6333 keypoints)](docs/images/figure_26.png)

**Rotated 90°**
![SIFT - Rotated 90° (5076 keypoints)](docs/images/figure_27.png)

**Brighter (alpha=1.5, beta=30)**
![SIFT - Brighter (8472 keypoints)](docs/images/figure_28.png)

**Darker (alpha=0.7, beta=-20)**
![SIFT - Darker (3738 keypoints)](docs/images/figure_29.png)

**Blurred (kernel=7x7)**
![SIFT - Blurred (2306 keypoints)](docs/images/figure_30.png)

**Noisy (std_dev=30)**
![SIFT - Noisy (8657 keypoints)](docs/images/figure_31.png)

## 💡 Key Insights

### Scale Invariance
- SIFT maintains consistent keypoint detection across different scales
- Harris shows dramatic changes with scaling, making it unsuitable for multi-scale analysis

### Rotation Invariance
- SIFT descriptors are computed relative to the dominant orientation
- Harris responses are orientation-dependent

### Feature Matching
- SIFT can reliably match features between different scales and rotations
- Harris corners alone cannot be matched reliably without additional descriptor computation

### Computational Complexity
- Harris is significantly faster to compute
- SIFT is slower but provides more reliable results
- Trade-off between speed and robustness should guide algorithm selection

## 🔧 How to Use

1. **Load Images**: The notebook automatically downloads the Kaggle dataset
2. **Detect Features**: Run cells to detect Harris corners and SIFT keypoints
3. **Visualize Results**: View generated plots showing detection results
4. **Analyze Robustness**: Apply transformations and observe algorithm behavior
5. **Compare Detectors**: Examine side-by-side comparisons in generated visualizations

## 📚 Learning Outcomes

After completing this project, you will understand:

- How corner detection algorithms work
- The mathematics behind Harris corner detection
- The concepts of scale and rotation invariance
- SIFT algorithm fundamentals
- Feature matching techniques
- Image transformation effects on detection algorithms
- When to use Harris vs SIFT in different applications

## 🎓 Applications

These techniques are fundamental for:
- **Object Recognition** - Detecting distinctive features for identification
- **Image Stitching** - Aligning multiple images
- **3D Reconstruction** - Building 3D models from 2D images
- **Video Tracking** - Following objects across frames
- **SLAM** (Simultaneous Localization and Mapping) - Robot navigation
- **Content-Based Image Retrieval** - Finding similar images

## 📖 References

- Harris, C., & Stephens, M. (1988). A Combined Corner and Edge Detector
- Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints
- OpenCV Documentation: https://docs.opencv.org/
- Computer Vision: Algorithms and Applications by Richard Szeliski

## 📄 Project Report

For detailed analysis, mathematical derivations, and extended results, see [report.pdf](report.pdf)

## 👨‍💼 Author

**Yusuf** - Semester 9 Computer Vision Project

## 📞 Support

For questions or issues, please refer to the detailed comments in the Jupyter notebook and the comprehensive report.

## 📜 License

This project is part of an academic course. Please use responsibly and provide attribution.

---

**Last Updated**: December 2025


