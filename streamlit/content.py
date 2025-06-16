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
We use Streamlit here and the end user can easily select a model, upload an image and get the prediction result.
"""

home_normal = """#### Class: normal"""
home_viral_pneumonia = """#### Class: viral_pneumonia"""
home_lung_opacity = """#### Class: lung_opacity"""
home_covid = """#### Class: COVID"""


modelisation_variables = """
| Variable          | Description       |
|-------------------|-------------------|
| name              | name of the model (the leading number reflects the sequence in our experimentation) |
| description       | brief description of the model |
| covid_f1          | COVID class f1 score |
| covid_precision   | COVID class precision score (high precision: few false positives |
| covid_recall      | COVID class recall score (high recall: few false negatives) |
| average_f1        | average f1 score across all classes |
| classes           | list of classes the model predicts |
| preprocessing     | preprocessing applied to the images within the model |
| data_augmentation | data augmentation applied during training |
| transfer_learning | transfer learning applied by using the pretrained parameters |
| fine_tuning       | fine-tuning applied by unfreezing the layers of the pretrained model |
| masked            | masked images used for training |
| dense_layers      | list of dense layers in the model |
| dropout           | dropout rate applied between the dense layers |
| epochs            | max number of epochs the model was trained |
| early_stopping    | number of epochs the model training stopped after no more learning |
| batch_size        | batch size used for training |
| optimizer         | optimizer used for training |
| learning_rate     | initial learning rate used for training, was reduced when plateau reached |
| loss_function     | loss function used for training |
| metrics           | metrics used for training |
| total_params         | total number of parameters in the model |
| trainable_params     | number of trainable parameters in the model |
| non_trainable_params | number of non-trainable parameters in the model |
"""

modelisation_summary_columns = [
    "name",
    "description",
    "covid_f1",
    "covid_precision",
    "covid_recall",
    "average_f1",
    "classes",
    "preprocessing",
    "data_augmentation",
    "transfer_learning",
    "fine_tuning",
    "masked",
    "dense_layers",
    "dropout",
    "epochs",
    "early_stopping",
    "batch_size",
    "optimizer",
    "learning_rate",
    "loss_function",
    "metrics",
    "total_params",
    "trainable_params",
    "non_trainable_params",
]

prediction_note = """
*Note: please remove any uploaded image before entering an image URL*
"""


prediction_or = """
<div style='text-align: center; padding-top: 2em;'>OR</div>
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
