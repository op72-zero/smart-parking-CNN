# fixed_predict.py - WITH RANDOM IMAGES
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

class ParkingSlotPredictor:
    def __init__(self, model_path):
        """Initialize parking slot predictor"""
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        img = cv2.resize(img, (224, 224))
        
        # Normalize pixel values to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict_single_image(self, image_path):
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_path)
            
            # Make prediction
            prediction = self.model.predict(processed_img, verbose=0)[0][0]
            
            print(f"DEBUG: Raw prediction value: {prediction}")
            
            if prediction > 0.5:
                class_name = "Empty"  
                confidence = float(prediction)
            else:
                class_name = "Occupied"  
                confidence = 1 - float(prediction)
                
            return class_name, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0
    
    def predict_with_visualization(self, image_path):
        """Predict and show result with image visualization"""
        class_name, confidence = self.predict_single_image(image_path)
        
        if class_name is None:
            return None, 0.0
        
        # Load image for display
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        
        # Set title based on prediction
        title_color = 'red' if class_name == 'Occupied' else 'green'
        plt.title(f'Prediction: {class_name}\nConfidence: {confidence:.2%}', 
                 fontsize=16, color=title_color, pad=20)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return class_name, confidence

def get_random_images(num_images=4):
    """Get random images from both classes"""
    test_images = []
    
    for class_folder in ['empty', 'occupied']:
        folder_path = os.path.join('data', 'raw', class_folder)
        if os.path.exists(folder_path):
            images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Get random samples instead of first ones
            if len(images) > num_images // 2:
                random_samples = random.sample(images, num_images // 2)
                test_images.extend(random_samples)
            else:
                test_images.extend(images)
    
    # Shuffle the final list
    random.shuffle(test_images)
    return test_images

def main():
    """Main function for demonstration"""
    # Find the latest model
    model_files = [f for f in os.listdir('models') if f.endswith('.h5')]
    
    if not model_files:
        print("No trained model found. Please train the model first.")
        return
    
    # Use the best model
    model_path = os.path.join('models', model_files[0])
    predictor = ParkingSlotPredictor(model_path)
    
    # Get random images each time
    test_images = get_random_images(num_images=4)
    
    if test_images:
        print("\n" + "="*50)
        print("PARKING SLOT CLASSIFICATION - RANDOM IMAGES")
        print("="*50)
        print(f"Testing on {len(test_images)} random images...")
        print("Close each image window to see the next prediction...")
        
        correct_predictions = 0
        total_predictions = 0
        
        for i, image_path in enumerate(test_images):
            print(f"\n--- Testing Image {i+1}: {os.path.basename(image_path)} ---")
            
            # Get expected class from filename
            expected = "Empty" if "empty" in image_path.lower() else "Occupied"
            print(f"Expected: {expected}")
            
            # Make prediction with visualization
            predicted, confidence = predictor.predict_with_visualization(image_path)
            
            if predicted:
                is_correct = (predicted == expected)
                result_str = "CORRECT" if is_correct else "WRONG"
                result_icon = "✅" if is_correct else "❌"
                
                print(f"Predicted: {predicted} ({confidence:.2%})")
                print(f"Result: {result_icon} {result_str}")
                
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
        
        # Summary
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        
        print(f"\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        print(f"Total Predictions: {total_predictions}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.1f}%")
        
        if accuracy == 100.0:
            print("ALL PREDICTIONS CORRECT! System is working perfectly!")
        else:
            print(" Some predictions incorrect. Check the class mapping.")
        print("="*50)
        
    else:
        print("No test images found.")

if __name__ == "__main__":
    main()