ds_general = """
## Demo, how a data scientist would use the MLOps solution

A data scientist usually develops a model and wants to put this model into production.
Hence the data scientest must be able to register the new model within the MLOps solution.
We use MLFlow here and the data scientist can easily register the new model, by
- authenticating himself as a registered datascientist
- calling the register_model endpoint on the ml_train_hub service

The implementation of this MLOps solution not only registers the model,
it actually holds a validation set and the metrics are calculated from this model automatically.
Also the model architecture is extracted and saved as parameters in MLFlow.

Once the model has been registered, it will be available to the end user.
"""

user_general = """
## Demo, how an end user would use the MLOps solution
An end user usually wants to use a model for prediction.
Hence the end user must be able to select a model and upload an image for prediction.
"""

about = """
## About
#### Contributors
- Benjamin Ries
- Markus Fuchs
- Philipp Würfel
- Vishal Desai

#### Thankyou
- to our project mentor **Kilyan Poilly** for his support and help.
- to **[DataScientest](https://www.datascientest.com/)** for the great training and the opportunity to work on this project.
"""
