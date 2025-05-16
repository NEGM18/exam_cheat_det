from ml_model import CheatingDetectionModel

def main():
    # Initialize the model
    model = CheatingDetectionModel()
    
    # Prepare dataset
    print("Preparing dataset...")
    X_train, X_val, y_train, y_val = model.prepare_dataset(
        cheating_dir="cheating_videos",
        normal_dir="normal_videos"
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Train the model
    print("Training model...")
    history = model.train(X_train, y_train, X_val, y_val)
    
    # Save the trained model
    print("Saving model...")
    model.save_model("cheating_detection_model.h5")
    print("Model saved successfully!")

if __name__ == "__main__":
    main() 