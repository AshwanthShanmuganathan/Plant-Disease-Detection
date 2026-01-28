from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from model import PlantDiseaseModel
import logging
import base64
import uuid
from datetime import datetime

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
app.config['ENV'] = 'development'
app.config['DEBUG'] = True

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize model
model = PlantDiseaseModel()

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a PNG or JPEG image'})
        
        # Create a unique filename with original extension
        original_extension = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}.{original_extension}"
        
        # Save the file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Make prediction with uploaded flag
        result = model.predict(filepath, source_type='uploaded')
        # Add image path to result with /static prefix
        result['image_path'] = f"/static/uploads/{unique_filename}"
        result['source'] = 'uploaded'
        
        # Log the prediction and image path
        logging.info(f"Uploaded image prediction for {unique_filename}: {result}")
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'Error during prediction: {str(e)}'})

@app.route('/capture', methods=['POST'])
def capture():
    try:
        # Get the base64 image data
        image_data = request.json.get('image', '').split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Save the image with a unique filename
        unique_filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        # Make prediction with captured flag
        result = model.predict(filepath, source_type='captured')
        result['image_path'] = f"/static/uploads/{unique_filename}"
        result['source'] = 'captured'
        
        # Log the prediction
        logging.info(f"Captured image prediction for {unique_filename}: {result}")
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error during capture: {str(e)}")
        return jsonify({'error': f'Error during capture: {str(e)}'})

if __name__ == '__main__':
    import socket
    def get_ip():
        try:
            # Get the computer's hostname
            hostname = socket.gethostname()
            # Get the IP address
            ip_address = socket.gethostbyname(hostname)
            return ip_address
        except:            
            return "0.0.0.0"
            
    try:
        port = 8000  # Try port 8000 instead of 5000
        ip = get_ip()
        
        print(f"\n🌿 Plant Disease Detection Server is running!")
        print(f"\nYou can access the application at:")
        print(f"* Local URL:     http://localhost:{port}")
        print(f"* Network URL:   http://{ip}:{port}")
        print(f"* Alternative:   http://127.0.0.1:{port}")
        print(f"\nShare this link with others on your network: http://{ip}:{port}")
        print("\n📝 Note: If accessing from another device on the network:")
        print("   - Make sure both devices are on the same network")
        print("   - Try any of the above URLs if one doesn't work")
        print("   - If links don't work, check your firewall settings")
        print("\n⚡ Press CTRL+C to quit the server")
        
        app.run(host='0.0.0.0', port=port, threaded=True, debug=True)
        
    except Exception as e:
        print(f"\n❌ Error starting server: {str(e)}")
        print("\nTrying alternative port 8080...")
        try:
            app.run(host='0.0.0.0', port=8080, threaded=True, debug=True)
        except Exception as e:
            print(f"\n❌ Error starting server on alternative port: {str(e)}")
            print("Please make sure no other applications are using ports 8000 and 8080")
