# sign-language-prediction
# Welcome to our ASL Predictor web application! 
## Webapp that uses Machine Learning to predict the sign language based on an image the user uploads or clicks from their webcam

The dataset used in this study is the Sign Language MNIST, which is a modified version of the classic MNIST dataset. This dataset contains images of American Sign Language (ASL) hand gestures, with each image representing one of the 26 letters of the alphabet (excluding J and Z due to gesture motions). ASL is a complete and natural language primarily used by deaf and hard-of-hearing individuals in North America.

Dataset: https://www.kaggle.com/datasets/datamunge/sign-language-mnist?datasetId=3258

List of files:
- Analysis_model.ipynb 
(Jupyter notebook containing the data analysis, preprocessing, model building, results, and code to save the model)
- main.py
(Contains code for the web app design, model load and integration, prediction, and to run the app)
- data folder
(Contains the training and testing set for the ML model)
- model folder
(Contains the saved CNN model as .pb files)
- modelh5.h5
(The saved CNN model in .h5 format)

If you want to run the code you can run it in Pycharms / terminal / jupyter notebook. The analysis_model.ipynb file shows how we trained the model to predict ASL images. If you want to run it, you can or you can just look at the outputs of the code to have an overall understanding of how the model works and how accurate it is. We also have a load function which loads and saves the model in a file. 
The main.py represents the web app where users can use this app to learn how to do ASL alphabets. There are multiple ways to run this file. You can execute the last cell in jupyter notebook, run the last segment in PyCharm, or enter the following command in terminal when you are in the correct directory: python3 main.py and it will output a localhost port address. Just click on the address and it should direct you to a page with our web app. 

Considerations:
- Make sure to have modelh5.h5 file in the same directory as your project when running the code as that represents the saved ML model for predicting ASL images. 
- Since I do not know the libraries that already exist in your computer, any missing libraries can be easily installed using the pip install command.



