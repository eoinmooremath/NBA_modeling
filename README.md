# NBA Fantasy Sports Optimization 🏀 💻

A sophisticated machine learning pipeline that leverages deep learning and optimization techniques to generate winning NBA DraftKings lineups. This project combines cutting-edge transformer architecture, real-time data integration, and mathematical optimization to create a powerful fantasy sports prediction system.

## Project Overview 🎯

This end-to-end system demonstrates expertise in:
- Deep Learning with Transformers
- Real-time Data Integration
- Mathematical Optimization
- Large-scale Data Processing
- Full-stack Development

## System Architecture 🏗️

### Historical Data Collection & Training
The foundation of the system is built on comprehensive NBA data:
- Massive dataset of 60,000+ NBA games (1946-2024)
- Rich player statistics and biographical information
- Advanced feature engineering and preprocessing
- Sophisticated data cleaning and normalization pipeline

### Deep Learning Engine 🧠
At the heart of the system is the SAINT (Self-Attention and Intersample Attention Transformer) architecture:
- Dual-attention mechanism captures complex basketball dynamics:
  - Row-wise self-attention models player-to-player interactions and team chemistry
  - Column-wise self-attention identifies player archetypes and statistical correlations
- Custom embedding pathways handle both categorical and continuous features
- Advanced regularization techniques prevent overfitting
- Original SAINT architecture introduced [here](https://github.com/somepago/saint?tab=readme-ov-file)


### Model Training 🎓
The SAINT model is trained on a comprehensive dataset of over 60,000 NBA games spanning from 1946 to 2024. This extensive training enables the model to recognize complex patterns in player performance and team dynamics.

### Daily Prediction Pipeline 📊
The system runs daily to generate optimal DraftKings lineups through a sophisticated three-stage process:

1. **Data Collection**
   - Daily collection of current player statistics from NBA.com
   - Retrieval of DraftKings contest structures and player salaries
   - Processing of today's draft groups and game matchups

2. **Performance Prediction**
   - For each game, the trained model predicts detailed player statistics
   - Predictions include points, rebounds, assists, and other key metrics
   - Conversion of predicted statistics into projected fantasy points

3. **Lineup Optimization**
   - For each DraftKings contest:
     - Mathematical optimization using linear programming solver PuLP
     - Maximizes total projected fantasy points
     - Enforces DraftKings' roster requirements
     - Maintains salary cap constraint ($50,000)
   - Outputs guaranteed optimal lineup selections
   - Print results and save them in .csv files.

4. **Betting**
   - Go to DraftKings.com and place your bets!

## Project Structure 📁

```
├── configs/            # Configuration and hyperparameters
├── data/              # Data management
│   ├── processed/     # Cleaned and engineered features
│   └── raw/          # Original data archives
├── models/            # Model architecture and checkpoints
├── predictions/       # Generated lineup outputs
└── src/              # Source code
    ├── data/         # Data processing and integration
    ├── models/       # Deep learning implementation
    └── prediction/   # Optimization engine
```

## Core Components 🔧

- `nba_scraper.py`: Robust data collection system
- `data_prep.py`: Advanced feature engineering pipeline
- `saint.py`: State-of-the-art transformer implementation
- `draft_kings.py`: Real-time DraftKings integration
- `predict_draft_groups.py`: Main prediction orchestrator
- `prediction_utils.py`: Mathematical optimization engine

## Usage 🚀

Train the deep learning model:
```bash
python src/models/run_training.py
```

Generate optimized lineups:
```bash
python src/prediction/predict_draft_groups.py
```

## Technical Stack 💪

- PyTorch for deep learning
- PuLP for mathematical optimization
- pandas & numpy for data processing
- requests & BeautifulSoup4 for web integration

## Dependencies 📚

- PyTorch
- PuLP
- pandas
- numpy
- requests
- BeautifulSoup4
