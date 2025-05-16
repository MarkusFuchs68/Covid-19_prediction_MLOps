import os
import shutil
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.data import AUTOTUNE
from PIL import Image
import random
from pathlib import Path
from sklearn.model_selection import train_test_split


def make_and_store_eval_dataset(folder_path, batch_size, output_folder):
    eval_ds_raw = image_dataset_from_directory(
        folder_path,
        label_mode="int",
        batch_size=batch_size,
        image_size=(299, 299), # keep the original image size
        validation_split=0.2,
        subset="validation",
        seed=42
    )
    class_names = eval_ds_raw.class_names
    eval_ds = eval_ds_raw.prefetch(buffer_size=AUTOTUNE)

    # Clear output folder if exists
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # Create subfolders for each class
    for class_name in class_names:
        os.makedirs(os.path.join(output_folder, class_name), exist_ok=True)

    # Save images as PNG
    image_count = 0
    for images, labels in eval_ds:
        for i in range(images.shape[0]):
            image_tensor = images[i].numpy().astype("uint8")
            label_name = class_names[labels[i].numpy()]
            save_path = os.path.join(output_folder, label_name, f"img_{image_count}.png")
            Image.fromarray(image_tensor).save(save_path, format="PNG")
            image_count += 1

    print(f"Saved {image_count} PNG images to '{output_folder}'")


def copy_validation_images_preserve_quality(source_dir, output_dir, val_ratio=0.2, seed=42):
    random.seed(seed)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # Clean output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Collect files per class
    class_to_files = {}
    for class_dir in source_dir.iterdir():
        if class_dir.is_dir():
            files = list(class_dir.glob("*.png"))
            if files:
                class_to_files[class_dir.name] = files

    # Stratified split and copy
    for class_name, file_list in class_to_files.items():
        train_files, val_files = train_test_split(
            file_list, test_size=val_ratio, random_state=seed
        )
        target_class_dir = output_dir / class_name
        target_class_dir.mkdir(parents=True, exist_ok=True)
        for file_path in val_files:
            shutil.copy(file_path, target_class_dir / file_path.name)

    print(f"Validation set copied to: {output_dir}")


if __name__ == "__main__":
    # Define the folder path and parameters
    folder_path = "data/original_dataset"
    output_folder = "data/eval_dataset"
    batch_size = 32

    # Call the function to create the evaluation dataset
    #copy_validation_images_preserve_quality(folder_path, output_folder)
    make_and_store_eval_dataset(folder_path, batch_size, output_folder)
    # The function will create the dataset and print the class names