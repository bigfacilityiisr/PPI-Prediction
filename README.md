
#Protein-Protein Interaction Prediction
This repository contains the model and code for predicting protein-protein interactions (PPIs) by inputting protein sequences.

The trained model is implemented using a Streamlit applet, which can be accessed through the following link: PPI Prediction App. Users can predict the interaction between two proteins by uploading their corresponding sequences.

##Dataset and Model Training
The model was trained on a dataset of 5,000 protein sequences retrieved from UniProt. Various machine learning algorithms were tested, and the best model was selected for deployment in the applet. The LightGBM (LGBM) algorithm demonstrated superior performance compared to Support Vector Machine (SVM) and XGBoost (XGB).

##Abstract of the Study
In this study, we evaluated the performance of machine learning techniques for predicting protein-protein interactions using sequential data. By employing diverse computational instruments and methods, our objective was to enhance the understanding of PPIs and assess the effectiveness of machine learning techniques in predicting these interactions.

A robust dataset was utilized to train and evaluate the model, yielding satisfactory performance metrics. Cross-validation methods were implemented to ensure that the model is applicable across various scenarios. The findings indicate that machine learning techniques can successfully predict PPIs, providing insights that may guide future research and biotechnological applications.

Getting Started
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/your-repository.git
Install the required packages: Make sure to install the necessary libraries to run the code. You can do this by running:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app: After installing the dependencies, you can run the app locally using:

bash
Copy code
streamlit run app.py









