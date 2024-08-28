# FundTapMLOps

FundTapMLOps is a machine learning operations project for FundTap, designed to streamline the process of training and deploying machine learning models.

## Prerequisites

- Docker
- Python 3.x
- pip

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/Seascape2010/FundTapMLOps.git
   cd FundTapMLOps
   ```

2. Set up Snowflake credentials:
   - Copy `.secret.example` and rename it to `.secret`
   - Open `.secret` and input your Snowflake credentials

3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running Locally

To train the model locally:

```
python src/train.py --dictionary "data/fundtap-data-dictionary.csv" \
--data "data/processed_data.csv" \
--output "model/" \
--n-trials 10
```

### Using Docker for Training

1. Build the Docker image for training:
   ```
   docker build -t fundtapmlops-train:latest -f training-dockerfile .
   ```

2. Run the Docker container for training:
   ```
   docker run -v $(pwd):/app/FundTapMLOps fundtapmlops-train:latest \
     --dictionary "/app/FundTapMLOps/data/fundtap-data-dictionary.csv" \
     --data "/app/FundTapMLOps/data/train.csv" \
     --output "/app/FundTapMLOps/model/" \
     --n-trials 10
   ```

### Using Docker for Inferencing

1. Build the Docker image for inferencing:
   ```
   docker build -t fundtapmlops-inference:latest -f inferencing-dockerfile .
   ```

2. Run the Docker container for inferencing:
   ```
   docker run -p 8080:8080 fundtapmlops-inference:latest
   ```

   The inference API will be available at `http://localhost:8080`.

## Docker Image Analysis

To view the sizes of individual layers in the Docker images:

```
docker history fundtapmlops-train:latest --human --format "{{.Size}}\t{{.CreatedBy}}"
docker history fundtapmlops-inference:latest --human --format "{{.Size}}\t{{.CreatedBy}}"
```

To get a summary of the total image sizes:

```
docker image ls fundtapmlops-train:latest
docker image ls fundtapmlops-inference:latest
```

## Project Structure

- `src/`: Source code for the ML model
- `data/`: Input data files
- `model/`: Output directory for trained models
- `inferencing/`: Code for model inferencing
- `training-dockerfile`: Instructions for building the training Docker image
- `inferencing-dockerfile`: Instructions for building the inferencing Docker image
- `requirements.txt`: Python dependencies

## API Usage

After running the inferencing Docker container, you can make predictions using the following endpoint:

- POST `/predict`

Example curl command:
```
curl -X POST "http://localhost:8080/predict" -H "Content-Type: application/json" -d '{"quote": 1000.0}'
```

Replace the input data with your actual features as required by the model.

## Development

For local development and testing, you can use Jupyter notebooks located in the `notebooks/` directory. Make sure to add the following code at the beginning of your notebooks to import from the `src` directory:

```python
import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_root)

# Now you can import from src
from src.preprocessing_helpers import preprocess_features
```