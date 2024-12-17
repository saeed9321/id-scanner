import cv2
import re
import os
import easyocr

def preprocess_image(image_path):
    """
    Preprocess the image to enhance text visibility.
    """
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return image, thresh

def extract_text_from_image(image):
    """
    Extract text from image using EasyOCR.
    """
    # Initialize EasyOCR reader for Arabic and English
    reader = easyocr.Reader(['ar', 'en'])
    
    # Extract text with additional parameters for better accuracy
    results = reader.readtext(
        image,
        paragraph=False,  # Treat each text block separately
        detail=1,
        contrast_ths=0.3,
        adjust_contrast=0.5,
        add_margin=0.1,
        width_ths=0.7,
        height_ths=0.7
    )
    
    # Combine all detected text
    full_text = '\n'.join([result[1] for result in results])
    return full_text, results

def extract_fields_from_text(text, raw_results):
    """
    Parse the extracted text to get Name, Date of Birth, and ID Number.
    Optimized for Omani ID cards.
    """
    name = None
    dob = None
    id_number = None

    # Patterns specific to Omani ID cards
    name_indicators = ['الاسم', 'اسم', 'Name', 'الاسم الكامل', 'Full Name']
    dob_indicators = ['تاريخ الميلاد', 'Date of Birth', 'DATE OF BIRTH']
    id_indicators = ['الرقم المدني', 'CIVIL NUMBER']

    # Regex patterns
    dob_patterns = [
        r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
        r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
    ]
    
    id_patterns = [
        r'\b\d{8}\b',         # 8 digits (Omani format)
        r'\b\d{8,10}\b',      # 8-10 digits (flexible match)
    ]

    # Process raw results with their positions
    sorted_results = sorted(raw_results, key=lambda x: (x[0][0][1], x[0][0][0]))  # Sort by y, then x
    
    # First pass: Look for exact matches with labels
    for result in raw_results:
        text_block = result[1].strip()
        
        # Look for ID number
        if any(indicator in text_block for indicator in id_indicators):
            # Check the same line for numbers
            numbers = re.findall(r'\d+', text_block)
            if numbers:
                id_number = numbers[0]
            else:
                # Look in adjacent text
                idx = raw_results.index(result)
                if idx + 1 < len(raw_results):
                    next_text = raw_results[idx + 1][1]
                    numbers = re.findall(r'\d+', next_text)
                    if numbers:
                        id_number = numbers[0]

        # Look for date of birth
        if any(indicator in text_block for indicator in dob_indicators):
            for pattern in dob_patterns:
                dob_match = re.search(pattern, text_block)
                if dob_match:
                    dob = dob_match.group()
                    break
            if not dob:
                # Check next line
                idx = raw_results.index(result)
                if idx + 1 < len(raw_results):
                    next_text = raw_results[idx + 1][1]
                    for pattern in dob_patterns:
                        dob_match = re.search(pattern, next_text)
                        if dob_match:
                            dob = dob_match.group()
                            break

    # Second pass: Direct pattern matching if not found
    if not id_number:
        for result in raw_results:
            text_block = result[1]
            for pattern in id_patterns:
                id_match = re.search(pattern, text_block)
                if id_match:
                    id_number = id_match.group()
                    break
            if id_number:
                break

    if not dob:
        for result in raw_results:
            text_block = result[1]
            for pattern in dob_patterns:
                dob_match = re.search(pattern, text_block)
                if dob_match:
                    dob = dob_match.group()
                    break
            if dob:
                break

    # Extract Name - look for Arabic text that appears to be a name
    for result in sorted_results:
        text_block = result[1].strip()
        # Look for Arabic text with 2-4 words
        if re.search(r'[\u0600-\u06FF]+', text_block):  # Contains Arabic
            words = text_block.split()
            if (2 <= len(words) <= 4 and 
                not any(char.isdigit() for char in text_block) and
                not any(indicator in text_block for indicator in name_indicators + dob_indicators + id_indicators)):
                name = text_block
                break

    return name, dob, id_number

def extract_face(image, output_face_path="extracted_face.jpg"):
    """
    Detect and extract face from the ID card using Haar cascades.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        cv2.imwrite(output_face_path, face)
        print(f"Face extracted and saved to {output_face_path}")
        return face

    print("No face detected in the image.")
    return None

def main(image_path):
    """
    Main function to extract ID card information.
    """
    print("Processing the ID card image...")

    # Preprocess the image
    image, thresh = preprocess_image(image_path)

    # Extract text
    text, raw_results = extract_text_from_image(image)
    print("Extracted Text:\n", text)

    # Parse fields
    name, dob, id_number = extract_fields_from_text(text, raw_results)
    print("\nParsed Information:")
    print(f"Name: {name if name else 'Not found'}")
    print(f"Date of Birth: {dob if dob else 'Not found'}")
    print(f"ID Number: {id_number if id_number else 'Not found'}")

    # Extract face
    extract_face(image)

    # Display the processed image
    cv2.imshow("Processed Image", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "id_card.jpg"  # Replace with the path to your ID card image
    if not os.path.exists(image_path):
        print(f"Image file '{image_path}' not found. Place an ID card image in the same directory.")
    else:
        main(image_path)
