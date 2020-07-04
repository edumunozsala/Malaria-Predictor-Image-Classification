# Malaria predictor: a image classification problem

## Problem description 
Malaria is a deadly, infectious, mosquito-borne disease caused by Plasmodium parasites that are transmitted by the bites of infected female Anopheles mosquitoes. There are five parasites that cause malaria, but two types —P. falciparum and P. vivax— cause the majority of the cases.
There are several methods that can be used for malaria detection and diagnosis. The paper on which our project is based, ["Pre-trained convolutional neural networks as feature extractors toward improved Malaria parasite detection in thin blood smear images,"](https://peerj.com/articles/4568/) by Rajaraman, et al., introduces some of the methods, including polymerase chain reaction (PCR) and rapid diagnostic tests (RDT). These two tests are typically used where high-quality microscopy services are not readily available. According to WHO protocol, diagnosis typically involves intensive examination of the blood smear at 100X magnification. Trained people manually count how many red blood cells contain parasites out of 5,000 cells.

Deep learning models, or more specifically convolutional neural networks (CNNs), have proven very effective in a wide variety of computer vision tasks.

## Content
This repository contains some files I build in the development of the Capstone Project in the Microsoft Professional Program in Artificial Intelligence in April - May 2019. 
- Malaria Infected Predictor - Preprocessing Train and evaluate a CNN: This notebooks follows all the steps in the ML process: load, preprocess and save the images and then create or define a convolutional neural network, apply data augmentation, train and evaluate the model. But in this notebook the model is very simple, its for demostration purposes only. 
- Malaria Infected Predictor- Image preprocessing: Load the images, rescale or normalized them and save them to disk in a friendly format for numpy arrays.
- Malaria Infected Predictor - Train and evaluate a CNN: Here we create a model with 3 Conv + pooling layers, train and evaluate.
- Malaria Infected Predictor - Train and evaluate a CNN and DataGenerator: Apply data augmentation techniques to our datasets to prevent overfitting.
- Malaria Infected Predictor - Train and evaluate a CNN Transfer Learning: Using a pre-trained model we create a classifier on top of it.
- Malaria Infected Predictor - Testing a predictor: When we create a model using the Azure ML notebooks, after training we test it prediction habilities using this notebook.
- Azure ML notebook: A group of notebooks and python scripts to train a CNN using the Azure machine learning services as a freame work for developing ml models.

## About the data
The data for our analysis comes from researchers at the Lister Hill National Center for Biomedical Communications (LHNCBC), part of the National Library of Medicine (NLM), who have carefully collected and annotated the publicly available dataset of healthy and infected blood smear images. They used Giemsa-stained thin blood smear slides from 150 P. falciparum-infected and 50 healthy patients, collected and photographed at Chittagong Medical College Hospital, Bangladesh. The smartphone's built-in camera acquired images of slides for each microscopic field of view. The images were manually annotated by an expert slide reader at the Mahidol-Oxford Tropical Medicine Research Unit in Bangkok, Thailand.

[Link to the data repository](https://ceb.nlm.nih.gov/repositories/malaria-datasets/)


## Contributing
If you find some bug or typo, please fixit and push it to be applied 

## License

These notebooks are under a public GNU License.