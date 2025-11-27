# Malicious URL Classifier

This project implements a machine learning model to classify URLs as either malicious or benign. The model is trained using a dataset provided in the `url.mat` file and additional data from the SVM-light files located in the `url_svmlight` folder.

## Project Structure

```
malicious-url-classifier
├── data
│   ├── url.mat
│   └── url_svmlight
│       └── (svm-light files .svm / .txt / .dat)
├── src
│   ├── train_model.py        # Script to train the SVM model
│   ├── predict.py            # Script to classify input URLs
│   ├── features.py           # Functions for feature extraction from URLs
│   └── utils.py              # Utility functions for data processing and evaluation
├── models
│   └── model.pkl             # Trained SVM model
├── notebooks
│   └── exploration.ipynb      # Jupyter notebook for exploratory data analysis
├── tests
│   └── test_pipeline.py       # Unit tests for the project
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd malicious-url-classifier
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure that the `data` folder contains the `url.mat` file and the `url_svmlight` folder with the necessary SVM-light files.

## Usage

### Training the Model

To train the model, run the following command:
```
python src/train_model.py
```
This will load the datasets, train the SVM model, and save the trained model as `model.pkl` in the `models` directory.

### Classifying URLs

To classify a URL, use the `predict.py` script. You can input a URL directly in the script or modify it to accept user input. Run the following command:
```
python src/predict.py
```

### Exploratory Data Analysis

For exploratory data analysis, open the Jupyter notebook:
```
jupyter notebook notebooks/exploration.ipynb
```

## Testing

To run the unit tests, execute:
```
python -m unittest discover -s tests
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.