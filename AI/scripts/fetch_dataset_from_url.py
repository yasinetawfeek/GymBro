import requests



def url_exists(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
    
def extract_dataset_name(url):
    """
    extract the dataset name from the url assuming that this url is from huggingface.co

    # https://huggingface.co/datasets/averrous/workout
    # https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset-v1
    # https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset-v1?library=datasets

    Targets:
    1 - averrous/workout
    2 - nvidia/Llama-Nemotron-Post-Training-Dataset-v1
    3 - nvidia/Llama-Nemotron-Post-Training-Dataset-v1
    """
    url_parts = [part for part in url.split("/datasets")]
    dataset_name = url_parts[-1].split("?")[0]
    dataset_name = dataset_name[1:len(dataset_name)]
    print(dataset_name)
    return dataset_name

def check_if_huggingface(url):
    if url.startswith("https://huggingface.co"):
        return True
    else:
        return False

# extract_dataset_name("https://huggingface.co/datasets/averrous/workout")
# extract_dataset_name("https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset-v1?library=datasets")
# extract_dataset_name("https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset-v1")
# extract_dataset_name("https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT")

# print(check_if_huggingface("https://husggingface.co/datasets/averrous/workout"))