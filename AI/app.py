# Add a health endpoint to the AI service
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200 