# Data Loaders

The **dataloaders** folder in this repository contains various utility scripts and classes related to data loading, preprocessing, and handling for the remote sensing image captioning project. Below is a brief overview of the contents of this folder:

## Contents

- `Data_loader.py`: This script provides a data loader class that facilitates loading and batching of image-caption pairs for training and validation.

- `extract.py`: This script includes functions for extracting relevant information from various data sources or formats.

- `__init__.py`: This file makes the `dataloaders` folder a Python package, allowing its modules to be imported in other scripts.

- `dataset.py`: Here, you'll find a dataset class that manages the dataset for training, validation, and testing. It helps in efficiently retrieving data samples.

- `graber.py`: This script might include functions to acquire or download data from external sources if needed.

- `JSONloaders.py`: The script provides functions to load data from JSON files. JSON files often store annotations and metadata associated with images and captions.

- `Target_token.py`: This script defines classes or functions to preprocess and manage target tokens (words or tokens in captions), possibly including tokenization and vocabulary management.

## Usage

To utilize the functionalities provided in the dataloaders folder, you can import the relevant classes and functions in your main project scripts. Here's an example of how to potentially use the data loader and dataset classes:

```python
from dataloaders.Data_loader import DataLoader
from dataloaders.dataset import ImageCaptionDataset

# Create a DataLoader instance for training
train_loader = DataLoader(dataset=ImageCaptionDataset(...), batch_size=..., shuffle=True)

# Iterate over batches during training
for batch_data in train_loader:
    images, captions = batch_data
    # Perform training steps

# Similar usage can be applied for validation and testing
```

Additionally, you can explore other scripts in this folder to see how they contribute to data preprocessing, extraction, and loading.

## Acknowledgments

The data loading and preprocessing components within the dataloaders folder have been developed with insights drawn from the domains of data management, image processing, and natural language processing.

For a deeper understanding and more context, feel free to delve into the code and documentation provided in the dataloaders directory.

## Contact

For any queries, suggestions, or contributions related to the dataloaders or other aspects of the project, don't hesitate to reach out to us at [project_email@example.com](mailto:project_email@example.com). Your involvement and feedback are highly appreciated!
