## DSP_Project
This repository is dedicated to the data science practice project titled 'Understanding Real and Fake Faces'.

## Dataset
The dataset consists of 1081 real images and three categories of a total of 960 fake images—easy, medium, and hard to detect.

Dataset: https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection

![Dataset Overview](Images/Example_images.png)

## Directory Structure

```
├── Augmentation_ds.ipynb          # For apply face croping and augmentation on images
├── CNN.py                         # Proposed CNN model
├── CNN_Feature_Extraction.ipynb   # For featuremap extraction from each images
├── Classification.py              # For classification using lazypredict
├── EDA.py                         # For exploratory data analysis
├── Eigen_faces.ipynb              # For generating eigen faces
├── LIME_Explainer.py              # XAI using LIME
├── SHAP_Explainer.py              # XAI using SHAP

``` 
## Methodology & Deliverables

1. **Workflow**
![Workflow](Images/Workflow.png)

2. **SHAP Explainer**
![SHAP Explainer](Images/SHAP.png)

3. **LIME Explainer**
![LIME Explainer](Images/LIME.png)


