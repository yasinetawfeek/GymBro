from flask import Flask, request, jsonify
# from .workout_classifer_model import train_model
from ai_model import train_model,predict

#train_model(training_dataset,testing_dataset)

app = Flask(__name__)
API_KEY ="job_hunting_ai_memory_leakage" #should be the same as the one in viewset.py in django app
# ALLOWED_IP="10.167.143.148" # Django App if running locally
ALLOWED_IP="172.18.0.1" #if running Docker
@app.route('/train_classify_workout', methods=['POST'])
def train_classify_workout():

    """
    API for training classify_workout model
    """
    print("request.remote_addr",request.remote_addr)
    if request.remote_addr != ALLOWED_IP:
        return {"error": "Unauthorized"}, 403
    
    try:
        data = request.get_json()
        # print("data",data)
        # if 'file' in request.files:
        #     print("yes")
        # else:
        #     print("no")
        # training_dataset = request.files['file']
        training_dataset = data.get("training_dataset")
        testing_dataset = data.get("testing_dataset")
        print("training_dataset",training_dataset)
        accuracy = train_model(training_dataset, testing_dataset)

        return jsonify({"status": "success", "accuracy": accuracy})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/predict_classify_workout', methods=['POST'])
def predict_classify_workout():
    print("request.remote_addr",request.remote_addr)
    if request.remote_addr != ALLOWED_IP:
        return {"error": "Unauthorized"}, 403
    try:
        data = request.get_json()
        data_to_predict = data.get("data_to_predict")
        workout_label = predict(data_to_predict)
        return jsonify({"status": "success", "workout_label": workout_label})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)