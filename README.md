# NBA Fantasy Predictor

A machine learning system for predicting NBA fantasy sports performance using neural networks.

## Project Overview

This project uses deep learning to predict NBA player fantasy performance. It consists of three main components:

1. **Data Collection**: Automated scraping of NBA statistics and DraftKings data
2. **Model Training**: Implementation of the SAINT neural network architecture for predictions
3. **Prediction**: Real-time predictions for DraftKings contests

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nba-fantasy-predictor.git
cd nba-fantasy-predictor

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. **Data Collection**:
```bash
python -m src.data.nba_scraper
```

2. **Model Training**:
```bash
python -m src.models.training
```

3. **Making Predictions**:
```bash
python -m src.prediction.predictor
```

## Project Structure

```
nba-fantasy-predictor/
├── data/              # Data storage
├── models/           # Saved models
├── notebooks/        # Jupyter notebooks
└── src/             # Source code
    ├── data/        # Data collection
    ├── models/      # Model architecture
    └── prediction/  # Prediction logic
```

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.