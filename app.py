from flask import Flask, request, jsonify
import os
import re
import requests

# Set up Hugging Face token
import requests

# Replace with your actual token
token = "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"
headers = {"Authorization": f"Bearer {token}"}
url = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"

response = requests.get(url, headers=headers)

if response.status_code == 200:
    print("Token is valid.")
else:
    print(f"Failed to validate token: {response.status_code} - {response.text}")

app = Flask(__name__)

def predict_html_control_type(fieldname):
    prompt = f"What is the HTML input type for '{fieldname}'? Reply with just the input type ."
       
    try:
        # Log the prompt for debugging
        print(f"Sending prompt to model: {prompt}")

        # Get prediction from the Hugging Face model using requests
        response = requests.post(url, headers=headers, json={"inputs": prompt})
        
        # Check if the response is valid
        response.raise_for_status()  # Raise an error for bad responses
        
        # Check the response format
        response_json = response.json()
        
        # Log the raw response for debugging
        print(f"Raw response: {response_json}")

        # If the response is a list, extract the first element
        if isinstance(response_json, list):
            prediction = response_json[0].get('generated_text', '').lower().strip() if isinstance(response_json[0], dict) else ''
        else:
            prediction = response_json.get('generated_text', '').lower().strip() if isinstance(response_json, dict) else ''

        # Log the prediction for debugging
        print(f"Received prediction: {prediction}")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response content: {response.text}")  # Log the response content
        prediction = ""
    except Exception as e:
        print(f"Model prediction failed for '{fieldname}': {e}")
        prediction = ""

    # Define input types and their associated keywords
    input_types = {
        'text': ['name', 'username', 'address', 'description', 'comments', 'feedback', 'notes'],
        'email': ['email', 'mail'],
        'password': ['password', 'pass'],
        'date': ['date', 'dob', 'birthday'],
        'datetime-local': ['datetime', 'appointment'],
        'number': ['age', 'salary', 'payment', 'amount', 'price', 'cost', 'fee', 'wage'],
        'tel': ['phone', 'tel', 'contact'],
        'checkbox': ['available', 'subscribe', 'terms', 'agree'],
        'radio': ['gender', 'sex'],
        'textarea': ['description', 'comments', 'feedback'],
        'color': ['color', 'background'],
        'file': ['file', 'upload', 'attachment'],
        'image': ['image', 'picture'],
        'calendar': ['month', 'year'],
        'range': ['range', 'slider'],
        'reset': ['reset', 'clear'],
        'search': ['search', 'query'],
        'submit': ['submit', 'send'],
        'time': ['time', 'appointment'],
        'url': ['url', 'link'],
        'button': ['button', 'action']
    }

    # Check the prediction for input types
    if prediction:
        for input_type in input_types.keys():
            if input_type in prediction:
                return f"<input type='{input_type}'>"

    # Fallback to keyword matching
    for input_type, keywords in input_types.items():
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', fieldname.lower()) for keyword in keywords):
            return f"<input type='{input_type}'>"

    return "<input type='text'>"  # Default to text input

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    fieldname = data.get('fieldname', '')
    if not fieldname:
        return jsonify({"error": "Field name is required"}), 400

    html_control_type = predict_html_control_type(fieldname)
    return jsonify({"fieldname": fieldname, "html_control_type": html_control_type})

if __name__ == '__main__':
    app.run(debug=True)