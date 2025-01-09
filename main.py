import cv2
import os
from predict import extract_and_predict
import torch
from train import create_model

def normalize_image(img, target_width=793):
    """Normalize image width to 793px while maintaining aspect ratio"""
    height, width = img.shape[:2]
    scale = target_width / width
    new_height = int(height * scale)
    return cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_CUBIC)

def calculate_row_height(row_number):
    """Calculate y-coordinate for a given row"""
    base_y = 75  # Starting y position
    row_height = 71  # Row spacing - this controls the gap between rows
    return int(row_height * row_number + base_y)

def cut_image(input_path, output_dir="cropped_images", target_size=(224, 224)):
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
        (37, 230),    # Slightly wider first column, moved left
        (275, 235),   # Second column unchanged
        (520, 245)    # Slightly wider third column
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
                
                # Draw rectangle on original image to verify cuts
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Add question number to debug image
                cv2.putText(img, str(question_num), (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            except Exception as e:
                print(f"Error processing cut {question_num}: {str(e)}")
                continue
    
    # Save debug image with rectangles
    cv2.imwrite(os.path.join(output_dir, "debug_cuts.png"), img)
    print("\nAll cuts completed. Check the 'cropped_images' folder.")
    
    return cut_paths, valid_questions

def process_image(image_path):
    # Cut the image
    print(f"\nProcessing: {image_path}")
    cut_paths, valid_questions = cut_image(image_path)
    
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
    for cut_path, question_num in zip(cut_paths, valid_questions):
        result = extract_and_predict(model, cut_path, device)
        if result.strip():  # Only add non-blank results
            results.append({
                "question": question_num,
                "answer": result
            })
            print(f"Cut {question_num}: {result}")
        else:
            print(f"Skipping cut {question_num} - no digits detected")
    
    # Save results to JSON file
    import json
    output_filename = os.path.splitext(image_path)[0] + "_results.json"
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {output_filename}")

def main():
    # Process each image in the current directory
    for filename in os.listdir():
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            process_image(filename)

if __name__ == "__main__":
    main()