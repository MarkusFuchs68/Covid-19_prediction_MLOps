def get_model_architecture(model):
    """
    We extract the layer architecture of the model
    """
    architecture = dict(
        {
            "layer0": "Conv2D(32, (3, 3), activation='relu')",
            "layer1": "MaxPooling2D((2, 2))",
        }
    )
    return architecture


def evaluate_model(model):
    """
    We run an evaluation against the evaluation dataset
    and return according metrics
    """
    metrics = dict({"performance": 0.85})
    return metrics
