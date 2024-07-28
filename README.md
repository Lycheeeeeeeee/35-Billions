
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
python src/train.py --dictionary "data/fundtap-data-dictionary.csv" --data "data/train.csv" --output "model/"
```

### Using Docker

1. Build the Docker image:
   ```
   docker build -t fundtapmlops:0.2 .
   ```

2. Run the Docker container:
   ```
   docker run -v $(pwd):/app/FundTapMLOps fundtapmlops:0.2 \
     --dictionary "/app/FundTapMLOps/data/fundtap-data-dictionary.csv" \
     --data "/app/FundTapMLOps/data/train.csv" \
     --output "/app/FundTapMLOps/model/"
   ```

## Docker Image Analysis

To view the sizes of individual layers in the Docker image:

```
docker history fundtapmlops:0.2 --human --format "{{.Size}}\t{{.CreatedBy}}"
```

To get a summary of the total image size:

```
docker image ls fundtapmlops:0.2
```

## Project Structure

- `src/`: Source code for the ML model
- `data/`: Input data files
- `model/`: Output directory for trained models
- `Dockerfile`: Instructions for building the Docker image
- `requirements.txt`: Python dependencies
