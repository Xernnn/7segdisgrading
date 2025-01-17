import cv2
import os
from predict import extract_and_predict
import torch
from train import create_model
import numpy as np
import sys

def normalize_image(img, target_width=793):
    """Normalize image width to 793px while maintaining aspect ratio"""
    height, width = img.shape[:2]
    scale = target_width / width
    new_height = int(height * scale)
    return cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_CUBIC)

def calculate_row_height(row_number):
    """Calculate y-coordinate for a given row"""
    base_y = 58  # Starting y position
    row_height = 68  # Row spacing - this controls the gap between rows
    return int(row_height * row_number + base_y)

def load_config(json_path="results.json"):
    """Load coordinates and answers from results.json"""
    import json
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert coordinates array to expected format
        coordinates = {
            'top_left': [float(data['coordinates'][0]['x']), float(data['coordinates'][0]['y'])],
            'top_right': [float(data['coordinates'][1]['x']), float(data['coordinates'][1]['y'])],
            'bottom_right': [float(data['coordinates'][2]['x']), float(data['coordinates'][2]['y'])],
            'bottom_left': [float(data['coordinates'][3]['x']), float(data['coordinates'][3]['y'])]
        }
        
        # Convert quiz_answers to expected format
        answers = {int(k): v for k, v in data.get('quiz_answers', {}).items()}
        
        return coordinates, answers
    except Exception as e:
        print(f"Error loading results.json: {str(e)}")
        return {}, {}

def cut_image(input_path, output_dir="cropped_images", target_size=(224, 224)):
    # Load correct answers
    correct_answers = load_correct_answers()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and normalize image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Could not read image: {input_path}")
        return
    
    # Normalize image width to 793px
    img = normalize_image(img)
    height, width = img.shape[:2]
    
    # Calculate number of possible rows based on image height
    max_rows = int((height - 85.25) / 77.6167) + 1  # Solve equation for x
    print(f"Image can fit {max_rows} rows")
    
    # Define the base widths and x-positions for each column (adjusted for better accuracy)
    columns = [
        (25, 230),    # First column
        (275, 235),   # Second column
        (520, 245)    # Third column
    ]
    
    cut_paths = []  # Store paths of cropped images
    valid_questions = []  # Store valid question numbers
    
    # Process each row and column
    for row in range(max_rows):
        y = calculate_row_height(row)
        h = 45
        
        # Process each column in the row
        for col, (x, w) in enumerate(columns):
            question_num = row * 3 + col + 1
            
            # Add small adjustments for specific positions
            if col == 2:  # Third column
                y_adjust = -2  # Move up slightly
                h_adjust = 2   # Make slightly taller
            else:
                y_adjust = 0
                h_adjust = 0
                
            # Skip if the cut would be outside the image
            if y + h > height:
                print(f"Skipping cut {question_num} - outside image bounds")
                continue
            
            # Extract with adjustments
            try:
                cropped = img[y + y_adjust:y + h + h_adjust, x:x+w]
                
                # Resize the cropped image
                cropped_resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_CUBIC)
                
                # Save the cropped image
                output_path = os.path.join(output_dir, f"cut_{question_num}.png")
                cv2.imwrite(output_path, cropped_resized)
                cut_paths.append(output_path)
                valid_questions.append(question_num)
                print(f"Saved cut {question_num} to {output_path}")
                
                # Draw rectangle on original image
                # Default color is white before prediction
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
                # Remove question number drawing
                
            except Exception as e:
                print(f"Error processing cut {question_num}: {str(e)}")
                continue
    
    return cut_paths, valid_questions, img  # Return img for later coloring

def process_image(image_path):
    # Cut the image
    print(f"\nProcessing: {image_path}")
    cut_paths, valid_questions, debug_img = cut_image(image_path)
    
    # Load correct answers
    correct_answers = load_correct_answers()
    
    # Set up model for prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = create_model()
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Process each cut
    print("\nPredicting numbers for each cut:")
    results = []
    
    # Define column positions for recoloring rectangles
    columns = [
        (25, 230),    # First column
        (275, 235),   # Second column
        (520, 245)    # Third column
    ]
    
    for cut_path, question_num in zip(cut_paths, valid_questions):
        result = extract_and_predict(model, cut_path, device)
        if result.strip():  # Only add non-blank results
            results.append({
                "question": question_num,
                "answer": result
            })
            print(f"Cut {question_num}: {result}")
            
            # Color the rectangle based on correctness
            row = (question_num - 1) // 3
            col = (question_num - 1) % 3
            y = calculate_row_height(row)
            x, w = columns[col]
            h = 45
            
            # Check if answer is correct
            correct_answer = correct_answers.get(question_num, "")
            
            # Only draw answer text if it exists in results.json and is not empty
            if correct_answer and question_num in correct_answers:
                # Draw black border for text
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    cv2.putText(debug_img, f"{correct_answer}", 
                              (x+70+dx, y-7+dy),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Draw colored text
                if result == correct_answer:
                    text_color = (0, 255, 0)  # Green for correct
                else:
                    text_color = (0, 0, 255)  # Red for wrong
                    
                cv2.putText(debug_img, f"{correct_answer}", 
                          (x+70, y-7),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            
            # Only draw rectangle if answer exists and is not "00000"
            if correct_answer and correct_answer != "00000":
                if result == correct_answer:
                    color = (0, 255, 0)  # Green for correct
                else:
                    color = (0, 0, 255)  # Red for wrong
                    
                # Draw colored rectangle
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 2)
            
        else:
            print(f"Skipping cut {question_num} - no digits detected")
    
    # Save final image outside cropped_images folder
    cv2.imwrite("final.png", debug_img)
    
    # Save results to JSON file
    import json
    output_filename = os.path.splitext(image_path)[0] + "_results.json"
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {output_filename}")

def extract_answer_section(image_path):
    """Extract the answer section from full test page using perspective transform"""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None
        
    # Load coordinates from results.json
    coordinates, _ = load_config()
    if not coordinates:
        print("Failed to load coordinates from results.json")
        return None
        
    # Define source points from coordinates
    src_points = np.float32([
        coordinates['top_left'],
        coordinates['top_right'],
        coordinates['bottom_right'],
        coordinates['bottom_left']
    ])
    
    # Define destination points (rectangle)
    width = int(src_points[1][0] - src_points[0][0])  # Use actual width from points
    height = int(src_points[2][1] - src_points[1][1]) # Use actual height from points
    
    dst_points = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])
    
    # Calculate perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply perspective transform
    answer_section = cv2.warpPerspective(img, matrix, (width, height))
    
    # Save the extracted section
    output_path = "extracted_section.png"
    cv2.imwrite(output_path, answer_section)
    
    return output_path

def process_full_page(image_path):
    """Process full test page"""
    # First extract the answer section
    answer_section_path = extract_answer_section(image_path)
    if answer_section_path is None:
        print("Failed to extract answer section")
        return
        
    # Process the extracted section as before
    process_image(answer_section_path)
    
    # Load coordinates for drawing
    coordinates, _ = load_config()
    if not coordinates:
        print("Failed to load coordinates for drawing")
        return
    
    # Read images
    original_img = cv2.imread(image_path)
    final_img = cv2.imread("final.png")
    
    if final_img is None:
        print("Failed to read processed image")
        return
    
    # Create copy of original for drawing
    result_img = original_img.copy()
    
    # Draw the region outline using coordinates from config
    pts = np.int32([
        coordinates['top_left'],
        coordinates['top_right'],
        coordinates['bottom_right'],
        coordinates['bottom_left']
    ])
    
    # Draw the region on the full page
    cv2.polylines(result_img, [pts], True, (0, 255, 0), 2)
    
    # Place the processed section back onto the full page
    # Calculate perspective transform matrix for placing back
    src_points = np.float32([
        [0, 0],
        [final_img.shape[1], 0],
        [final_img.shape[1], final_img.shape[0]],
        [0, final_img.shape[0]]
    ])
    
    dst_points = np.float32([
        [9.43074, 898.9375],    # Top left
        [949.71454, 898.9093],  # Top right
        [949.5916, 1194.6383],  # Bottom right
        [9.491907, 1194.481]    # Bottom left
    ])
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    result_section = cv2.warpPerspective(final_img, matrix, (original_img.shape[1], original_img.shape[0]))
    
    # Blend the result back into the original image
    mask = np.zeros(original_img.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], (255, 255, 255))
    mask_inv = cv2.bitwise_not(mask)
    
    result_img = cv2.bitwise_and(result_img, mask_inv)
    result_img = cv2.bitwise_or(result_img, result_section)
    
    # Save the final full page result
    cv2.imwrite("full_page_result.png", result_img)
    print("Full page result saved as full_page_result.png")

def load_correct_answers(json_path="results.json"):
    """Load correct answers from results.json"""
    _, answers = load_config(json_path)
    return answers

def main(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        return
        
    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        print("Error: File must be an image (PNG, JPG, or JPEG)")
        return
        
    # Process the image
    process_full_page(image_path)

if __name__ == "__main__":
    # When run directly, use command line argument
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
    else:
        main(sys.argv[1])