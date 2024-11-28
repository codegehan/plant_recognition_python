import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import shutil
import json
import time
from PIL import Image

class PlantClassifier:
    def __init__(self):
        self.dataset_path = 'dataset/training'
        os.makedirs(self.dataset_path, exist_ok=True)
        self.unknown_threshold = 0.80
        self.class_mapping_file = 'dataset/class_mapping.json'
        self.model_file = 'plant_classifier_model.keras'
        self.load_class_mapping()
        
        # Try to load existing model
        if os.path.exists(self.model_file) and self._get_plant_classes():
            try:
                print("\nLoading existing model...")
                self.model = tf.keras.models.load_model(self.model_file)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print("Creating new model...")
                self.model = self._build_model()
        else:
            print("\nNo existing model found. Creating new model...")
            self.model = self._build_model()
        
    def load_class_mapping(self):
        try:
            with open(self.class_mapping_file, 'r') as f:
                self.class_mapping = json.load(f)
        except FileNotFoundError:
            self.class_mapping = {}
            self.save_class_mapping()
            
    def save_class_mapping(self):
        with open(self.class_mapping_file, 'w') as f:
            json.dump(self.class_mapping, f, indent=4)
    
    def _build_model(self):
        # Get number of classes
        num_classes = len(self._get_plant_classes())
        if num_classes == 0:
            num_classes = 1  # Minimum one class

        model = models.Sequential([
            # Input layer - note the shape is (30, 64, 3)
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(30, 64, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.02)),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\nModel created with {num_classes} output classes")
        model.summary()
        return model
    
    def _get_plant_classes(self):
        """Get list of plant classes from dataset directory"""
        if not os.path.exists(self.dataset_path):
            return []
        classes = [d for d in sorted(os.listdir(self.dataset_path)) 
                  if os.path.isdir(os.path.join(self.dataset_path, d))]
        return classes
    
    def train(self, epochs=10, batch_size=32):
        classes = self._get_plant_classes()
        if not classes:
            print("No classes to train on!")
            return None
        
        print(f"\nTraining model on classes: {classes}")
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )

        train_generator = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=(30, 64),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=(30, 64),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=True
        )

        print("\nClass mapping:", train_generator.class_indices)
        
        # Save class indices to ensure consistent ordering
        self.class_mapping = train_generator.class_indices
        self.save_class_mapping()

        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            verbose=1
        )
        
        # Save the trained model in Keras format
        self.model.save(self.model_file)
        print("\nModel saved successfully")
        return history
    
    def classify_plant(self, image_path):
        try:
            # Check if we have any trained classes
            classes = self._get_plant_classes()
            if not classes:
                print("\nNo plants in database yet.")
                return self._handle_unknown_plant(image_path)
            
            # Make prediction
            predictions = self._predict(image_path)
            if predictions is None:
                return self._handle_unknown_plant(image_path)
            
            # Get prediction details
            confidence = float(np.max(predictions[0]))
            predicted_index = np.argmax(predictions[0])
            
            # Print all predictions for debugging
            print("\nPrediction confidences:")
            for class_name, idx in self.class_mapping.items():
                print(f"{class_name}: {predictions[0][idx]:.2%}")
            
            # If confidence is too low, treat as unknown
            if confidence < self.unknown_threshold:
                print(f"\nConfidence too low ({confidence:.2%})")
                return self._handle_unknown_plant(image_path)
            
            predicted_class = list(self.class_mapping.keys())[predicted_index]
            print(f"\nPlant detected: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
            return (predicted_class, confidence)
            
        except Exception as e:
            print(f"Error classifying image: {str(e)}")
            return self._handle_unknown_plant(image_path)
    
    def _handle_unknown_plant(self, image_path):
        if not self._get_plant_classes():
            print("\nThis will be the first plant in the database.")
        else:
            print("\nThis appears to be a new type of plant.")
        
        print("Would you like to add it to the database? (y/n)")
        
        if input().lower() == 'y':
            print("\nWhat type of plant is this?")
            plant_name = input().strip().lower()
            
            # Create new class directory if it doesn't exist
            class_dir = os.path.join(self.dataset_path, plant_name)
            os.makedirs(class_dir, exist_ok=True)
            
            try:
                # Save multiple versions of the image with different augmentations
                with Image.open(image_path) as img:
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    
                    # Save original
                    timestamp = int(time.time())
                    new_filename = f"{timestamp}.jpg"
                    new_path = os.path.join(class_dir, new_filename)
                    resized = img.resize((64, 30), Image.Resampling.LANCZOS)
                    resized.save(new_path, 'JPEG', quality=95)
                    
                    # Save flipped version
                    flipped = resized.transpose(Image.FLIP_LEFT_RIGHT)
                    flipped.save(os.path.join(class_dir, f"{timestamp}_flip.jpg"), 'JPEG', quality=95)
                    
                    # Save rotated versions
                    for angle in [-15, 15]:
                        rotated = resized.rotate(angle, expand=False)
                        rotated.save(os.path.join(class_dir, f"{timestamp}_rot{angle}.jpg"), 'JPEG', quality=95)
                
                # Update class mapping
                if plant_name not in self.class_mapping:
                    self.class_mapping[plant_name] = len(self.class_mapping)
                    self.save_class_mapping()
                
                print(f"\nSuccessfully added to database as: {plant_name}")
                
                # Rebuild and retrain model
                print("\nRebuilding and retraining model...")
                self.model = self._build_model()
                self.train()
                print("Model training complete!")
                
                return (plant_name, 0.0)
                
            except Exception as e:
                print(f"\nError saving image: {str(e)}")
                if os.path.exists(class_dir) and not os.listdir(class_dir):
                    os.rmdir(class_dir)  # Clean up empty directory
                return ("unknown", 0.0)
        
        print("\nImage was not added to the database.")
        return ("unknown", 0.0)

    def _predict(self, image_path):
        try:
            # Load and convert image if necessary
            with Image.open(image_path) as img:
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                # Ensure correct dimensions (width=64, height=30)
                img = img.resize((64, 30), Image.Resampling.LANCZOS)
                
                # Convert to numpy array with correct shape
                img_array = np.array(img)
                
                # Verify shape
                if img_array.shape != (30, 64, 3):
                    img_array = img_array.transpose(1, 0, 2)
                
                # Add batch dimension and normalize
                img_array = np.expand_dims(img_array, 0)
                img_array = img_array.astype('float32') / 255.0
                
                return self.model.predict(img_array, verbose=0)
                
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None