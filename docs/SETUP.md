# Setup Guide

## Installation

### 1. Clone Repository
```bash
git clone <repo-url>
cd Erdos-traffic-prediction
```

### 2. Create Conda Environment
```bash
conda env create -f config/environment.yml
conda activate kafka
```

### 3. Install Additional Dependencies (if needed)
```bash
pip install -r config/requirements.txt
```

### 4. Configure Environment Variables
```bash
cp config/.env.template .env
# Edit .env and add your AZ511 API key
```

## Data Setup

### Collect AZ511 Event Data
```bash
python database/az511.py
```
This creates `database/az511.db` with event records.

### INRIX Traffic Data
- INRIX data is proprietary and requires a license
- Place raw CSV files in `database/inrix-traffic-speed/I10-and-I17-1year/`
- Required files:
  - `I10-and-I17-1year.csv`
  - `TMC_Identification.csv`

## Running the Pipeline

See [Quick Start](../README.md#quick-start) in the main README.

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError`:
```bash
# Make sure you're in the project root directory
cd /path/to/Erdos-traffic-prediction

# Activate environment
conda activate kafka

# Check python path
python -c "import sys; print(sys.path)"
```

### Missing torch/pytorch
```bash
conda install pytorch torchvision torchaudio -c pytorch
```

### CUDA/GPU Issues
If training is slow, ensure PyTorch is using your GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory
Reduce batch size or use CPU:
```bash
# In training scripts, set:
# --batch-size 32  (default is 64)
# --device cpu
```
