
from fetch_dataset_from_url import url_exists, extract_dataset_name, check_if_huggingface
from mediapipe_format_dataset import mediapipe_format_dataset_handler
from workout_classifer_model import train_workout_and_evaluate
from muscle_group_classifer import train_muscle_group_and_evaluate
from mediapipe_handler import MediaPipeHandler
import os

def download_dataset(url):
    if url_exists(url) and check_if_huggingface(url):
        dataset_name = extract_dataset_name(url)
        print(f"Dataset name: {dataset_name}")
    else:
        print("Invalid URL or not a Hugging Face dataset.")
        return  False
    
    # Assuming the dataset is in a format that can be loaded directly
    if mediapipe_format_dataset_handler(dataset_name):
        print("Dataset loaded successfully.")
        return True
    else:
        print("Failed to load dataset.")
        return False
    

def train_workout_classifier():
    """
    Assuming that dataset has been stored and is ready to be loaded
    """
    mediapipe_model = MediaPipeHandler()
    training_dataset=mediapipe_model.read_csv_to_pd(os.path.join("data", "train_new.csv"))
    testing_dataset=mediapipe_model.read_csv_to_pd(os.path.join("data", "test_new.csv"))
    return train_workout_and_evaluate(training_dataset,testing_dataset)

def train_muscle_group_classifier():
    """
    Assuming that dataset has been stored and is ready to be loaded
    """
    mediapipe_model = MediaPipeHandler()

    training_dataset=mediapipe_model.read_csv_to_pd(os.path.join("data", "train_new.csv"))
    testing_dataset=mediapipe_model.read_csv_to_pd(os.path.join("data", "test_new.csv"))
    return train_muscle_group_and_evaluate(training_dataset,testing_dataset)

def train_displacement_classifier():
    """
    Assuming that dataset has been stored and is ready to be loaded
    """
    mediapipe_model = MediaPipeHandler()

    training_dataset=mediapipe_model.read_csv_to_pd("data\\train_new.csv")
    testing_dataset=mediapipe_model.read_csv_to_pd("data\\test_new.csv")
    # return train_displacement_and_evaluate(training_dataset,testing_dataset)

def predict(x):
    return "Push Up"




# download_dataset("https://huggingface.co/datasets/averrous/workout")
# train_workout_classifier()
# train_muscle_group_classifier()
# train_displacement_classifier()