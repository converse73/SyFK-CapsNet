# Maize (Zea mays L.) Yield Estimation Using an Improved FastKAN-based Deep Learning Model Coupled with Multi-source Remote Sensing Data


### Peer Review Special Note
This repository is exclusively for the peer review stage and is only accessible to reviewers. After the review process, it will be converted to a public repository for academic reference. The repository contains all source codes, datasets, pre-trained weights, and experimental output results related to the paper, which can directly reproduce the paper's experiments.

---

## 1. Paper Information

**Title:** Maize (Zea mays L.) Yield Estimation Using an Improved FastKAN-based Deep Learning Model Coupled with Multisource Remote Sensing Data

**Authors:** Jian Li, Junrui Kang, Jian Lu, Hongkun Fu, Weilin Yu, Weijian Zhang, Zheng Li, Xinglei Lin, Baoqi Liu, Hengxu Guan, Jiawei Zhao, Zhihan Liu

**Affiliations:**
- a. College of Information Technology, Jilin Agricultural University, Changchun 130118, China
- b. College of Resources and Environment, Jilin Agricultural University, Changchun 130118, China
- c. College of Agronomy, Jilin Agricultural University, Changchun 130118, China
- d. Northeast Institute of Geography and Agroecology, Chinese Academy of Sciences, Changchun 130102, China

**Corresponding Author:** Jian Li (lijian@jlau.edu.cn)

---

## 2. Project Introduction

Maize (Zea mays L.) is a pivotal industrial and bioenergy crop. Precise yield estimation is crucial for optimizing bio-based value chains and regional biomass resource allocation. Traditional deep learning architectures often struggle to preserve the structural integrity of complex environmental signals and capture non-monotonic crop responses to thresholds.

To address this, this project proposes **SyFK-CapsNet**, a novel deep learning architecture integrating:
- Transformer-LSTM for deep temporal dynamics modeling
- Capsule Networks for structured multi-dimensional feature aggregation
- FastKAN modules for precise nonlinear mapping

### Key Achievements
- **Northeast China Validation**: Achieved optimal performance on official statistics (\(R^{2}=0.8037\), \(RMSE = 458.9 ~kg/ha\))
- **Field Measurement Validation**: Independent field measurements (2022-2024) confirmed fine-scale robustness
- **Cross-Regional Generalization**: US Corn Belt validation (2013-2023) maintained an 11-year average \(R^{2}\) of 0.7808, peaking at \(R^{2}=0.8372\)
- **Model Interpretability**: SHAP analysis demystified predictive logic; Monte Carlo Dropout provided pixel-level uncertainty maps

This project provides a highly accurate, spatially generalizable, and transparent paradigm for large-scale industrial crop monitoring.

---

## 3. Environment Configuration

This section details the environment required to run the code, ensuring reviewers can quickly build the environment and reproduce experiments.

### 3.1 Basic Environment
- **OS Platform**: Windows 11 (AMD64)
- **Python Version**: 3.12.3
- **PyTorch Version**: 2.3.1+cu121
- **CUDA Version**: 12.1 (CUDA Available: True)
- **Active GPU**: NVIDIA GeForce RTX 4070 SUPER

### 3.2 Dependencies
**Core Dependencies:**
- numpy: 1.26.4
- pandas: 2.2.2
- scikit-learn: 1.5.0

**Specialized Libraries:**
- fast-kan: Installed (Successfully imported)
  - Source: https://github.com/ZiyaoLi/fast-kan

### 3.3 Environment Installation

**Method 1: Install via virtual environment (recommended)**

```bash
# Create and activate virtual environment (Windows system)
python -m venv .venv
D:\pytorch_test\pythonProject\.venv\Scripts\activate.bat

# Install all dependencies
pip install numpy==1.26.4 pandas==2.2.2 scikit-learn==1.5.0 torch==2.3.1+cu121
pip install git+https://github.com/ZiyaoLi/fast-kan.git
```

## 4. Data Preparation

> **Important Note**: The dataset of Northeast China is currently being used in our two subsequent related submitted research papers and cannot be made public for the time being. In addition, the field measured data is difficult to obtain. It will be made public later when it is no longer used in our related research.

To ensure the reproducibility of our research, we provide:
- Validation dataset of the SyFK-CapsNet model in the US Corn Belt region (including the US sample dataset)
- Complete source code of the SyFK-CapsNet model
- Pre-trained weight files for US validation (2013-2023)
- TIF file of the corn yield pixel map obtained from field measured data

For the complete acquisition of the original dataset, please refer to the following sections in our paper:
- Section 2.2: Maize yield data and farmland data
- Section 2.3: Remote sensing and meteorological data
- Section 2.4: Dataset preprocessing
- Section 5.3: Spatial Generalization and Robustness Validation in the US Corn Belt

---

## 5. Pre-trained Weights

### 5.1 Weight Storage Path
```plaintext
Pre-training weight/                # Saved model weights and outputs
├── models_SyFK-CapsNet_2013(USA_Statistical_Data)/
├── models_SyFK-CapsNet_2014(USA_Statistical_Data)/
├── ...
└── models_SyFK-CapsNet_2023(USA_Statistical_Data)/
    ├── SyKCABModel_R2_0.8146.pth          # Saved model weights
    ├── SyKCABModel_R2_0.8146_params.json  # Saved hyperparameters
    └── test_predictions_R2_0.8146.csv     # Exported test results
```

### 5.2 Weight Description and Testing
- The pre-trained weights are obtained by training the SyFK-CapsNet model on the US Corn Belt statistical data of each year
- Highest \(R^2\) achieved: 0.8372
- Each weight has corresponding hyperparameters stored in `.json` file
- Test prediction results are stored in `.csv` file for direct result verification

**Test Operation:**
```bash
# Run the evaluation script to test the pre-trained weights
python D:\pytorch_test\SyFK-CapsNet\code\evaluate.py --config D:\pytorch_test\SyFK-CapsNet\code\config.py --pretrained_path "D:\pytorch_test\SyFK-CapsNet\Pre-training weight\models_SyFK-CapsNet_2023(USA_Statistical_Data)\SyKCABModel_R2_0.8146.pth"
```

> **Note**: Replace `[Your pre-trained weight file path]` with the actual path, e.g., `D:\pytorch_test\SyFK-CapsNet\Pre-training weight\models_SyFK-CapsNet_2023(USA_Statistical_Data)\SyKCABModel_R2_0.8146.pth`

---

## 6. Repository Structure

```plaintext
SyFK-CapsNet/                           # Root directory of the project
├── .venv/                              # Python virtual environment (auto-generated)
├── code/                               # Core source code directory
│   ├── config.py                       # Hyperparameters and global settings
│   ├── dataset.py                      # Data loading and multi-source data alignment
│   ├── evaluate.py                     # Standalone evaluation and test script
│   ├── layers.py                       # Custom layers (FastKAN, Capsule Network components)
│   ├── main.py                         # Main entry for training and validation
│   ├── models.py                       # SyFK-CapsNet model main architecture
│   ├── train.py                        # Training loop and evaluation metric logic
│   └── utils.py                        # Auxiliary utility functions
│
├── Data/                               # Raw data and processed datasets
│
├── Pixel map of corn yield/            # Spatial pixel map of corn yield (TIF format)
│
├── Pre-training weight/                # Saved model weights and experimental outputs
│   ├── models_SyFK-CapsNet_2013(USA_Statistical_Data)/
│   ├── models_SyFK-CapsNet_2014(USA_Statistical_Data)/
│   ├── ...
│   └── models_SyFK-CapsNet_2023(USA_Statistical_Data)/
│       ├── SyKCABModel_R2_0.8146.pth          # Trained model weights
│       ├── SyKCABModel_R2_0.8146_params.json  # Corresponding hyperparameter records
│       └── test_predictions_R2_0.8146.csv     # Test set prediction comparison results
│
└── README.md                           # Project documentation
```

## 7. Citation

If you use the code, data, or weights of this project, please cite our paper:

```bibtex
@article{SyFK-CapsNet-Maize-Yield,
  title={Maize (Zea mays L.) Yield Estimation Using an Improved FastKAN-based Deep Learning Model Coupled with Multisource Remote Sensing Data},
  author={Li, Jian and Kang, Junrui and Lu, Jian and Fu, Hongkun and Yu, Weilin and Zhang, Weijian and Li, Zheng and Lin, Xinglei and Liu, Baoqi and Guan, Hengxu and Zhao, Jiawei and Liu, Zhihan},
  journal={[Journal Name]},
  year={[Publication Year]},
  volume={[Volume]},
  number={[Issue]},
  pages={[Pages]}
}
```

## 8. License

This project is licensed under the MIT License. For details, see the [LICENSE](LICENSE) file.

---

## 9. Contact Information

If you have any questions during the experiment reproduction process, please contact:
- **Jian Li**: lijian@jlau.edu.cn

We will respond to your inquiries in a timely manner.