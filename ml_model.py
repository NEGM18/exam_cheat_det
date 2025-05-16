import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
import os

class CheatingDetectionModel:
    def __init__(self, input_shape=(64, 64, 3)):
        self.input_shape = input_shape
        self.model = self._create_model()
        
    def _create_model(self):
        inputs = layers.Input(shape=self.input_shape)
        
        x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def preprocess_frame(self, frame, target_size=(64, 64)):
        frame = cv2.resize(frame, target_size)
        frame = frame / 255.0
        return frame
    
    def preprocess_video(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Error: Could not open video {video_path}")
                return None
            
            frames = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_rate = max(1, total_frames // 30)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_rate == 0:
                    processed_frame = self.preprocess_frame(frame)
                    frames.append(processed_frame)
                
                frame_count += 1
                
                if frame_count % 30 == 0:
                    print(f"Processing frame {frame_count}/{total_frames}", end='\r')
            
            cap.release()
            print(f"\n✅ Processed {len(frames)} frames from {os.path.basename(video_path)}")
            return np.array(frames)
        except Exception as e:
            print(f"❌ Error processing video {video_path}: {str(e)}")
            return None
    
    def prepare_dataset(self, cheating_dir, normal_dir):
        X_cheating = []
        X_normal = []
        
        print("\nProcessing cheating videos...")
        for video_file in os.listdir(cheating_dir):
            if video_file.endswith(('.mp4', '.avi')):
                print(f"\nProcessing {video_file}")
                video_path = os.path.join(cheating_dir, video_file)
                frames = self.preprocess_video(video_path)
                if frames is not None:
                    X_cheating.extend(frames)
        
        print("\nProcessing normal videos...")
        for video_file in os.listdir(normal_dir):
            if video_file.endswith(('.mp4', '.avi')):
                print(f"\nProcessing {video_file}")
                video_path = os.path.join(normal_dir, video_file)
                frames = self.preprocess_video(video_path)
                if frames is not None:
                    X_normal.extend(frames)
        
        if not X_cheating or not X_normal:
            raise ValueError("No valid frames extracted from videos!")
        
        print(f"\nExtracted {len(X_cheating)} frames from cheating videos")
        print(f"Extracted {len(X_normal)} frames from normal videos")
        
        X = np.array(X_cheating + X_normal)
        y = np.array([1] * len(X_cheating) + [0] * len(X_normal))
        
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001
        )
        
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr]
        )
        return history
    
    def predict_video(self, video_path):
        frames = self.preprocess_video(video_path)
        predictions = self.model.predict(frames)
        return np.mean(predictions)
    
    def save_model(self, path):
        self.model.save(path)
    
    def load_model(self, path):
        self.model = models.load_model(path)

if __name__ == "__main__":
    model = CheatingDetectionModel()
    
    X_train, X_val, y_train, y_val = model.prepare_dataset(
        cheating_dir="path/to/cheating/videos",
        normal_dir="path/to/normal/videos"
    )
    
    history = model.train(X_train, y_train, X_val, y_val)
    
    model.save_model("cheating_detection_model.h5")
