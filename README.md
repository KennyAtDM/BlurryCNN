# BlurryCNN

This repository contains a blurry image classifier project, which uses a convolutional neural network (CNN) to classify images as blurry or non-blurry.

## Dataset

The dataset used for this project is from [Kwentar's blur dataset](https://github.com/Kwentar/blur_dataset).

## Project Structure

```
Blurry_image_classifier/
│
├── blur_dataset/               # Contains dataset
├── output/                     # Saved models
├── inference/                  # experiments and visualizations
├── annotate.py
├── dataset.py                  # Data preprocessing scripts
├── model.py                    # Model architecture
├── train.py                    # Training script
├── inference.py                # Evaluation script
├── environment.yml             # Conda environment file
├── requirements.txt            # Python packages required
├── README.md                   # Project README file
└── .gitignore                  # Git ignore file
```

## Setup

### Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- Python 3.x

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/KennyAtDM/BlurryCNN.git
    cd Blurry_image_classifier
    ```

2. Create and activate the conda environment:

    ```sh
    conda env create -f environment.yml
    conda activate blurry_image_classifier
    ```

3. Install additional dependencies:

    ```sh
    pip install -r requirements.txt
    ```

## Usage


### Training

Train the model using the training script:

```sh
python train.py --epoch EPOCH
```

### Evaluation

Evaluate the model using the evaluation script:

```sh
python inference.py [-h] --checkpoint CHECKPOINT
```

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Dataset: [Kwentar's blur dataset](https://github.com/Kwentar/blur_dataset)

