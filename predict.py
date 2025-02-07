import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from train.train.train import create_model
import os

def predict_digit(model, image_path, device):
    # Match the exact preprocessing used in training
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        # Only return prediction if confidence is above threshold
        if confidence.item() > 0.7:
            return predicted.item()
        else:
            return None  # Return None for low confidence predictions

def clean_prediction(result):
    """Clean up prediction based on rules:
    1. Max 5 characters TOTAL (including minus and comma)
    2. Only one '-' allowed and only at start (remove if found elsewhere)
    3. Only one ',' allowed (not at start/end)
    4. If we have 5 digits, prioritize digits over comma
    """
    # Update mapping to match training
    LABEL_MAP = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
        5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: '-', 11: ','  # Make sure these match your training labels
    }
    
    # Remove any whitespace
    result = result.strip()
    
    # Initialize clean result
    cleaned = []
    has_minus = False
    has_comma = False
    digit_count = 0
    
    # Check for invalid comma positions
    if result.startswith(',') or result.endswith(','):
        result = result.replace(',', '')
    
    # First pass: check for minus at start
    if result.startswith('-'):
        has_minus = True
        cleaned.append('-')
        result = result[1:]
    
    # Remove any other minus signs
    result = result.replace('-', '')
    
    # Count digits first
    digits_only = [c for c in result if c.isdigit()]
    
    # If we have 5 or more digits, use only digits
    if len(digits_only) >= 5:
        cleaned_result = (''.join(cleaned) if has_minus else '') + ''.join(digits_only[:5])
        return cleaned_result
    
    # Otherwise process normally
    for char in result:
        # Stop if we have 5 characters total
        if len(cleaned) >= 5:
            break
            
        if char.isdigit():
            cleaned.append(char)
        elif char == ',' and not has_comma and len(cleaned) < 4:  # Need room for at least one more digit
            has_comma = True
            cleaned.append(char)
    
    # Join the cleaned result
    cleaned_result = ''.join(cleaned)
    
    # Pad with zeros if needed
    while len(cleaned_result) < 5:
        if cleaned_result.startswith('-'):
            # Insert zero after minus sign
            cleaned_result = '-' + '0' + cleaned_result[1:]
        else:
            # Add zero at start
            cleaned_result = '0' + cleaned_result
    
    return cleaned_result

def extract_and_predict(model, image_path, device):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return ""
        
    # Convert to grayscale and apply thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    # Filter and sort components
    valid_components = []
    min_height = height // 2.5  # Adjust minimum height threshold
    
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Skip very small components
        if area < 20:
            continue
            
        # Skip components that touch the edges
        if x <= 2 or (x + w) >= (width - 2):
            continue
            
        # Classify component type based on size and shape
        if h >= min_height:
            if w < width * 0.1:  # Very thin - likely '1'
                valid_components.append((x, y, w, h, 'one'))
            else:  # Normal digit
                valid_components.append((x, y, w, h, 'digit'))
        else:  # Small component
            if w > width * 0.1:  # Wide enough to be dash
                valid_components.append((x, y, w, h, 'dash'))
            else:  # Potential comma
                valid_components.append((x, y, w, h, 'comma'))
    
    # Sort components left to right
    valid_components.sort(key=lambda x: x[0])
    
    # Process components
    result = ""
    for x, y, w, h, comp_type in valid_components:
        try:
            if comp_type == 'dash':
                result += '-'
            elif comp_type == 'comma':
                result += ','
            elif comp_type in ['one', 'digit']:
                # Extract digit with padding
                pad = 2
                roi = gray[max(0, y-pad):min(height, y+h+pad), 
                          max(0, x-pad):min(width, x+w+pad)]
                
                # Save and predict
                temp_path = f"temp_digit.png"
                cv2.imwrite(temp_path, roi)
                
                digit = predict_digit(model, temp_path, device)
                if digit is not None:  # Only add if confidence is high enough
                    result += str(digit)
                
                os.remove(temp_path)
                
        except Exception as e:
            print(f"Error processing component: {str(e)}")
            continue
    
    # Clean the prediction
    result = clean_prediction(result)
    return result

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = create_model()
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Test on images
    test_folder = "test_images"
    for filename in os.listdir(test_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_folder, filename)
            result = extract_and_predict(model, image_path, device)
            print(f"File: {filename}, Predicted: {result}")

if __name__ == "__main__":
    main() 