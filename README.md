
# Crop Disease Detector

This Plant Disease Detector app uses "microsoft/swinv2-tiny-patch4-window8-256" which is a pre-trained image classification model from Hugging Face, that was then finetuned on combined dataset of "New Plant Diseases" and "Tea_Leaf_Disease" from kaggle.

The dataset used for training includes 92 thousand images of plant leaves labeled with different diseases categorized into 44 classes. Each image is annotated with the type of disease or if the plant is healthy.

The highest training accuracy achieved was 99.7546% and the test accuracy was 99.72%.


# Data set and model

Data set links: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
                https://www.kaggle.com/datasets/saikatdatta1994/tea-leaf-disease



Model is deployed on hugging face, which can accessed below.(While running the streamlit app, model would be automatically downloaded)

Model link: https://huggingface.co/Abhiram4/PlantDiseaseDetectorSwinv2


# Steps to use the app

## 1. Install Required Packages

These packages need to be installed using terminal

```pip install streamlit transformers Pillow```

If any package is missing in host system, you can install the missing one individually by using:

```pip install package-name```



## 2. Run the Streamlit App
To start the Streamlit app on your local host, run the following command in the terminal:

```streamlit run app.py```

