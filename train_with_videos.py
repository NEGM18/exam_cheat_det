from ml_model import CheatingDetectionModel
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    model = CheatingDetectionModel()
    
    BASE_DIR = "D:\\projects\\exam_cheat_det"
    cheating_dir = os.path.join(BASE_DIR, "cheating_videos")
    normal_dir = os.path.join(BASE_DIR, "normal_videos")
    
    print("\n=== Training Configuration ===")
    print(f"Looking for videos in:")
    print(f"Cheating videos: {cheating_dir}")
    print(f"Normal videos: {normal_dir}")
    
    print("\n=== Found Videos ===")
    cheating_videos = [v for v in os.listdir(cheating_dir) if v.endswith(('.mp4', '.avi'))]
    normal_videos = [v for v in os.listdir(normal_dir) if v.endswith(('.mp4', '.avi'))]
    
    print("Cheating videos:")
    for video in cheating_videos:
        print(f"- {video}")
    
    print("\nNormal videos:")
    for video in normal_videos:
        print(f"- {video}")
    
    if not cheating_videos or not normal_videos:
        print("\n❌ Error: No videos found in one or both directories!")
        print("Please record some videos first using record_video.py")
        return
    
    print("\n=== Preparing Dataset ===")
    print("This may take a few minutes...")
    X_train, X_val, y_train, y_val = model.prepare_dataset(
        cheating_dir=cheating_dir,
        normal_dir=normal_dir
    )
    
    print(f"\nDataset prepared:")
    print(f"- Training samples: {X_train.shape[0]}")
    print(f"- Validation samples: {X_val.shape[0]}")
    print(f"- Cheating samples: {sum(y_train == 1)}")
    print(f"- Normal samples: {sum(y_train == 0)}")
    
    if len(X_train) < 10:
        print("\n⚠️ Warning: Very few training samples!")
        print("Consider recording more videos for better results.")
        proceed = input("Do you want to continue training? (y/n): ")
        if proceed.lower() != 'y':
            return
    
    print("\n=== Training Model ===")
    print("Training will take several minutes...")
    history = model.train(X_train, y_train, X_val, y_val)
    
    print("\n=== Plotting Training History ===")
    plot_training_history(history)
    
    print("\n=== Evaluating Model ===")
    y_pred = model.model.predict(X_val)
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_classes))
    
    print("\n=== Plotting Confusion Matrix ===")
    plot_confusion_matrix(y_val, y_pred_classes)
    
    accuracy = np.mean(y_pred_classes == y_val)
    print(f"\nValidation Accuracy: {accuracy:.4f}")
    
    print("\n=== Saving Model ===")
    model_path = os.path.join(BASE_DIR, "cheating_detection_model.h5")
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")
    
    print("\n=== Training Complete ===")
    print("Check 'training_history.png' for training curves")
    print("Check 'confusion_matrix.png' for confusion matrix")
    print("\nYou can now use this model in video_monitor.py for real-time detection!")

if __name__ == "__main__":
    main() 