# CardNumber-DeepOCR

Iranian Credit card and National card Deep Optical Character Recognition (OCR) using Keras.

Dataset is synthesized using OpenCV.


## Table of Contents

1. [Dataset](#dataset)
2. [Model](#model)
3. [How to run](#how-to-run)
4. [Results](#results)
5. [App Demo](#app-demo)


## Dataset

### Iranian Credit Card (Different Banks)
Mellat Bank            |  Melli Bank
:-------------------------:|:-------------------------:
![Mellat-Bank](Assets/Mellat_Bank.jpg)  |  ![Melli-Bank](Assets/Melli_Bank.jpg)

Synthesized credit cards images path: Datasets/train_train_synthesized.zip inside Credit folder

### Iranian National card
Old ones            |  New ones
:-------------------------:|:-------------------------:
![National-Old](Assets/National_Old.jpg)  |  ![National-New](Assets/National_New.jpg)

Synthesized national cards images path: Datasets/train_train_synthesized.zip inside National folder


## Model

1. Synthesizing data and pre-processing [[Notebook]](Notebooks/Dataset.ipynb)
2. Classify the image (Credit or National) [[Notebook]](Notebooks/National_Credit_Classifier.ipynb)
3. Feed images to OCR network [[Notebook]](Notebooks/OCR_Network.ipynb)
4. Wrapping the pipeline [[Notebook]](Notebooks/E2E_Wrapper.ipynb)

### Classifier
<img src="Assets/Classifier_Model.png" alt="Classifier-Model" width="200"/>

### OCR
<img src="Assets/OCR_Model.png" alt="OCR-Model" width="200"/>


## How to run

For testing open ```E2E_Wrapper.ipynb``` inside Google Colab or open ```e2e_wrapper.py```

### API's:
Predict on a single image
```bash
predict_single_path(path, label=None)
```

Predict on a single image and show image, ground truth
```bash
single_file_show(path, label)
```

Predict on all images inside a folder path and report some metrics:
```bash
predict_folder(folder_path)
```


## Results

Credit cards            |  National cards
:-------------------------:|:-------------------------:
![Result-Credit](Assets/Result_Credit.jpg)  |  ![Result-National](Assets/Result_National.jpg)


## App Demo

https://user-images.githubusercontent.com/45814362/193408829-fdc9a889-09e6-4c61-a59e-a878efd00c2d.mp4
