from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import script  # your existing script.py file

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file and allowed_file(file.filename):
            # Secure the filename and save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image using your existing script
            image, thresh = script.preprocess_image(filepath)
            text, raw_results = script.extract_text_from_image(image)
            name, dob, id_number = script.extract_fields_from_text(text, raw_results)
            
            # Extract face and save it
            face_filename = f"face_{filename}"
            face_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
            script.extract_face(image, face_path)
            
            # Prepare results
            results = {
                'name': name if name else 'Not found',
                'dob': dob if dob else 'Not found',
                'id_number': id_number if id_number else 'Not found',
                'original_image': f'uploads/{filename}',
                'face_image': f'uploads/{face_filename}' if os.path.exists(face_path) else None
            }
            
            return render_template('index.html', results=results)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True) 