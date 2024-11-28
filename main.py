import os
import random
import string
from plant_classifier import PlantClassifier
import json

def accept_image_file(file_path):
    """
    Accept and validate an image file
    """
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            return False
            
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Validate extension
        if file_ext not in valid_extensions:
            print(f"Error: Invalid file format. Supported formats: {', '.join(valid_extensions)}")
            return False
            
        # Verify it's a valid image
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                img.verify()
            
            # Initialize classifier and classify image
            classifier = PlantClassifier()
            result = classifier.classify_plant(file_path)
            
            if result:
                plant_name, confidence = result
                if confidence == 0.0:
                    print(f"New plant type added: {plant_name}")
                else:
                    print(f"Plant classified as {plant_name} with {confidence:.2%} confidence")
            return True
            
        except Exception as e:
            print(f"Error: File is not a valid image - {str(e)}")
            return False
            
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return False


accept_image_file("banana.jpg")