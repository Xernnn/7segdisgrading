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
    base_y = 58  # Starting y position
    row_height = 68  # Row spacing - this controls the gap between rows
    return int(row_height * row_number + base_y)

def load_correct_answers(json_path="results.json"):
    """Load correct answers from results.json"""
    import json
    try:
        with open(json_path, 'r') as f:
            answers = json.load(f)
        # Convert to dictionary for easier lookup
        return {item["question"]: item["answer"] for item in answers}
    except Exception as e:
        print(f"Error loading results.json: {str(e)}")
        return {}

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
            if not correct_answer:  # No answer in results.json
                color = (255, 0, 0)  # Blue for no answer found
            elif result == correct_answer:
                color = (0, 255, 0)  # Green for correct
            else:
                color = (0, 0, 255)  # Red for wrong
                
            # Draw colored rectangle
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 2)
            
            # Add correct answer above the rectangle (only if answer exists)
            if correct_answer:
                cv2.putText(debug_img, f"{correct_answer}", (x+70, y-7), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black text
            
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

def main():
    # Process each image in the current directory
    for filename in os.listdir():
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            process_image(filename)

if __name__ == "__main__":
    main()