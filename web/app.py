"""
Web Interface for CycleGAN Image Translation
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Flask web application for CycleGAN image translation.
"""

from flask import Flask, render_template, request, jsonify, send_file
import torch
from PIL import Image
import io
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.networks import define_G
from torchvision import transforms
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['RESULTS_FOLDER'] = './results_web'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global model variables
netG_A2B = None
netG_B2A = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path, direction='AtoB'):
    """
    Load CycleGAN model.
    
    Args:
        model_path: Path to model checkpoint
        direction: Translation direction
    """
    global netG_A2B, netG_B2A
    
    if direction == 'AtoB' or netG_A2B is None:
        netG_A2B = define_G(3, 3, 64, 'resnet_9blocks', 'instance', False, 'normal', 0.02, 
                            [0] if torch.cuda.is_available() else [])
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            if isinstance(netG_A2B, torch.nn.DataParallel):
                netG_A2B = netG_A2B.module
            netG_A2B.load_state_dict(state_dict)
        netG_A2B.to(device)
        netG_A2B.eval()
    
    if direction == 'BtoA' or netG_B2A is None:
        netG_B2A = define_G(3, 3, 64, 'resnet_9blocks', 'instance', False, 'normal', 0.02,
                            [0] if torch.cuda.is_available() else [])
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            if isinstance(netG_B2A, torch.nn.DataParallel):
                netG_B2A = netG_B2A.module
            netG_B2A.load_state_dict(state_dict)
        netG_B2A.to(device)
        netG_B2A.eval()


def preprocess_image(image):
    """
    Preprocess image for CycleGAN.
    
    Args:
        image: PIL Image
        
    Returns:
        Preprocessed tensor
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0).to(device)


def tensor_to_image(tensor):
    """
    Convert tensor to PIL Image.
    
    Args:
        tensor: Image tensor
        
    Returns:
        PIL Image
    """
    tensor = tensor.squeeze(0).cpu()
    tensor = (tensor + 1) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    to_pil = transforms.ToPILImage()
    return to_pil(tensor)


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def translate():
    """
    Translate an image.
    
    Returns:
        JSON response with translated image
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        direction = request.form.get('direction', 'AtoB')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Load and preprocess image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image_tensor = preprocess_image(image)
        
        # Load model if needed
        model_path = request.form.get('model_path', None)
        load_model(model_path, direction)
        
        # Translate
        with torch.no_grad():
            if direction == 'AtoB':
                translated = netG_A2B(image_tensor)
            else:
                translated = netG_B2A(image_tensor)
        
        # Convert to image
        result_image = tensor_to_image(translated)
        
        # Convert to base64
        img_io = io.BytesIO()
        result_image.save(img_io, 'PNG')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_base64}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'cuda_available': torch.cuda.is_available()
    })


if __name__ == '__main__':
    print("CycleGAN Web Interface")
    print("Author: RSK World")
    print("Website: https://rskworld.in")
    print("Email: help@rskworld.in")
    print("Phone: +91 93305 39277")
    print("\nStarting server...")
    app.run(debug=True, host='0.0.0.0', port=5000)

