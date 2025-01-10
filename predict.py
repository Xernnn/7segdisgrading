import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from train import create_model
import os

def predict_digit(model, image_path, device):
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    
    # Prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        
    return predicted.item()

def clean_prediction(result):
    """Clean up prediction based on rules:
    1. Max 5 characters TOTAL (including minus and comma)
    2. Only one '-' allowed and only at start (remove if found elsewhere)
    3. Only one ',' allowed (not at start/end)
    4. If we have 5 digits, prioritize digits over comma
    """
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
    """Extract digits using connected components and predict each digit"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return ""
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Debug images
    debug_img = img.copy()
    debug_binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    # Filter and sort components
    valid_components = []
    comma_components = []  # Store potential comma components
    min_height = height // 3
    expected_width = width // 10
    
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        aspect_ratio = h/w if w > 0 else 0
        
        # Calculate component properties
        roi = gray[y:y+h, x:x+w]
        avg_intensity = np.mean(roi)
        fill_ratio = area / (w * h)
        
        # Skip components that touch the left or right edges
        if x <= 2 or (x + w) >= (width - 2):
            continue
            
        # Skip very thin vertical lines (likely grid lines)
        if w <= 3 and h > height * 0.5:
            continue
        
        # Use size to determine component type
        if h >= min_height * 1.3:  # Tall enough to be a digit
            if w <= expected_width * 0.4 and not (x <= 5 or x >= width-5):  # Very thin - must be '1'
                component_type = 'one'
                valid_components.append((x, y, w, h, area, component_type))
            else:  # Normal digit
                component_type = 'digit'
                valid_components.append((x, y, w, h, area, component_type))
        else:  # Small component
            if w > expected_width/2:  # Wide enough to be dash
                component_type = 'dash'
                valid_components.append((x, y, w, h, area, component_type))
            else:  # Potential comma
                comma_components.append((x, y, w, h, area, 'comma'))
    
    # If we have comma candidates, select the one with largest area
    if comma_components:
        largest_comma = max(comma_components, key=lambda x: x[4])  # x[4] is area
        valid_components.append(largest_comma)
    
    # Sort all components left to right
    valid_components.sort(key=lambda x: x[0])
    
    # Process components
    result = ""
    for i, (x, y, w, h, area, comp_type) in enumerate(valid_components):
        try:
            if comp_type == 'dash':
                result += '-'
                label = '-'
            elif comp_type == 'comma':
                result += ','
                label = ','
            elif comp_type == 'one':
                result += '1'
                label = '1'
            else:
                # Process other digits
                pad = 5
                x1 = max(0, x - pad)
                x2 = min(width, x + w + pad)
                y1 = max(0, y - pad)
                y2 = min(height, y + h + pad)
                
                char_img = gray[y1:y2, x1:x2]
                char_img = cv2.resize(char_img, (28, 28))
                
                # Save temporary image
                temp_path = f"temp_digit_{i}.png"
                cv2.imwrite(temp_path, char_img)
                
                # Predict only if not already classified as '1'
                digit = predict_digit(model, temp_path, device)
                result += str(digit)
                label = str(digit)
                
                # Clean up temp file
                os.remove(temp_path)
            
            # Add label to debug image
            cv2.putText(debug_img, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
        except Exception as e:
            print(f"Error processing component {i}: {str(e)}")
            continue
    
    # Clean the prediction according to rules
    result = clean_prediction(result)
    
    # Save debug images
    debug_dir = "debug_predictions"
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, f"binary_{os.path.basename(image_path)}"), binary)
    cv2.imwrite(os.path.join(debug_dir, f"prediction_{os.path.basename(image_path)}"), debug_img)
    cv2.imwrite(os.path.join(debug_dir, f"debug_{os.path.basename(image_path)}"), debug_binary)
    
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