# ActiveCMR Project

## Description


ActiveCMR is a research project focused on applying Active Learning techniques to Cardiac Magnetic Resonance (CMR) image segmentation. The goal is to improve data efficiency and model performance by selecting the most informative MRI slices for annotation. The project incorporates uncertainty estimation and adaptive data acquisition strategies to guide active sampling during training.


## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mmmmm-w/ActiveCMR.git
    cd ActiveCMR
    ```
2.  **Create a conda environment (recommended):**
    ```bash
    conda create -n cmr python=3.10 -y
    conda activate cmr
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

You can download from https://data.mendeley.com/datasets/pw87p286yx/1 and unzip `Dataset.zip` into this directory:

```bash
unzip Dataset.zip -d Dataset/
```

## Usage

**Demo**
```
python demo/demo.py
```

**Training**
```
python scripts/train_cvae.py
```
You may adjust configurations in the training script.

