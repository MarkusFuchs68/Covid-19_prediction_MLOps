import io
import json
import logging
import time

from tensorflow.keras.utils import image_dataset_from_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_model_architecture(model):
    """
    We extract the layer architecture of the model
    """
    # Redirect sys.stdout to capture model.summary() output
    stream = io.StringIO()

    # Redirect Keras summary output to a variable
    model.summary(
        print_fn=lambda x: stream.write(x + "\n")
    )  # Execute the model summary and capture it into the string buffer
    summary_str = stream.getvalue()  # Get the entire summary as a string

    # Parse summary output
    lines = summary_str.split("\n")  # Split the summary string into individual lines
    data = []  # We'll store parsed lines in this list
    for line in lines[
        2:-4
    ]:  # Parse each line (skipping the first two and the last two lines which are headers/footers)
        parts = [
            x for x in line.split("│") if x
        ]  # Split by │ and remove empty elements
        if len(parts) >= 3:
            # Extract layer name, output shape, and number of parameters
            layer_nametype = parts[0].lstrip().rstrip()
            output_shape = parts[1].lstrip().rstrip()
            param_count = parts[2].lstrip().rstrip()
            if (
                len(layer_nametype.strip()) > 0
                or len(output_shape.strip()) > 0
                or len(param_count.strip()) > 0
            ):  # Make sure all parts are non-empty before storing
                data.append([layer_nametype, output_shape, param_count])

    # The data list holds lists of model architecture data
    # The architecture data is: "Layer Name (type)", "Output Shape", "Param Count"
    # Convert it to a pretty formatted json object
    architecture = {
        "layer_"
        + str(index): {
            "layer_name": layer[0],
            "output_shape": layer[1],
            "param_count": layer[2],
        }
        for index, layer in enumerate(data)
    }

    # Return the data in pretty format json
    return json.dumps(architecture, indent=2)


def evaluate_model(model, class_names, max_num):
    """
    We run an evaluation against the evaluation dataset and return according metrics
    """
    try:
        # Read and resize the images to the models expected input shape
        input_shape = model.input_shape[1:3]  # Expected image size (height, width)
        dataset = read_and_resize_evaluation_dataset(class_names, input_shape)
        img_num = 0
        for batch in dataset:
            images, _ = batch
            img_num += images.shape[0]

        # Run the predictions on the evaluation dataset and take time for performance measurements
        logger.info(f"Starting predictions on dataset with {img_num} images...")
        start_time = time.time()
        y_true, y_pred = get_predictions_and_labels(model, dataset, img_num, max_num)
        duration = time.time() - start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        pred_time = f"{minutes}:{seconds:02d}"
        logger.info(f"Finished predictions on dataset in {pred_time} min:sec")

        # Calculate metrics manually without scikit-learn
        metrics = calculate_metrics(y_true, y_pred, class_names)
        return metrics  # this not in json format, since MLFlow expects a native dict object

    except Exception as e:
        logger.error(f"Error during evaluating the model metrics: {e}")
        raise  # and forward the exception


def read_and_resize_evaluation_dataset(class_names, input_shape=(224, 224)):
    # Load the entire dataset for validation
    dataset = image_dataset_from_directory(
        directory="data",
        class_names=class_names,  # only class names to include by model
        label_mode="int",  # for sparse_categorical_entropy training
        batch_size=8,  # Use defined batch size, rather small for saving system RAM
        image_size=input_shape,  # Resize input images
    )

    # Optionally, you can cache and prefetch for performance, but takes high RAM usage!
    # dataset = dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def get_predictions_and_labels(model, dataset, img_num, max_num):
    # Get true labels and predictions from the test dataset
    y_true = []  # List to hold true labels
    y_pred = []  # List to hold predicted labels

    img_count = 0
    for images, labels in dataset:

        # Get the model's predictions
        preds = model.predict(images, verbose=0)  # Predict class probabilities
        # Get the predicted labels (argmax)
        y_pred.extend(
            [int(list(p).index(max(p))) for p in preds]
        )  # Convert probabilities to class index (highest probability)
        # Get the true labels
        y_true.extend(labels.numpy())

        logger.debug(f"Predicted {len(y_true)} evaluation images out of {img_num}")

        if max_num > 0:
            img_count += len(y_true)
            if img_count > max_num:
                logger.debug(
                    f"Stopped predicting images at {img_count} images due to max_num {max_num}"
                )
                break

    return y_true, y_pred


def calculate_metrics(y_true, y_pred, class_names):
    # Calculate metrics natively
    metrics = {}
    # Convert to lists if not already
    y_true = list(y_true)
    y_pred = list(y_pred)
    total = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    metrics["accuracy"] = float(correct) / total if total > 0 else 0.0

    metrics["precision"] = {}
    metrics["recall"] = {}
    metrics["f1_score"] = {}

    for idx, class_name in enumerate(class_names):
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == idx and yp == idx)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != idx and yp == idx)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == idx and yp != idx)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics["precision"][class_name] = precision
        metrics["recall"][class_name] = recall
        metrics["f1_score"][class_name] = f1

    logger.info("METRICS:", metrics)
    print(metrics)
    return metrics


# For debugging
if __name__ == "__main__":
    print("try functions from debugpy here")
