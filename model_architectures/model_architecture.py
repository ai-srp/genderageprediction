import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

logger = logging.getLogger(__name__)

class ModelArchitectures:
    @staticmethod
    def simple_cnn(input_shape=(128, 128, 3)):
        """Simple CNN architecture"""
        logger.info("Building Simple CNN architecture")
        
        # Shared feature extraction layers
        inputs = keras.Input(shape=input_shape, name='input_image')
        
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        
        # Age prediction branch
        age_branch = layers.Dense(128, activation='relu', name='age_dense1')(x)
        age_branch = layers.Dropout(0.3)(age_branch)
        age_output = layers.Dense(1, activation='linear', name='age_output')(age_branch)
        
        # Gender prediction branch
        gender_branch = layers.Dense(64, activation='relu', name='gender_dense1')(x)
        gender_branch = layers.Dropout(0.3)(gender_branch)
        gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')(gender_branch)
        
        model = keras.Model(inputs=inputs, outputs=[age_output, gender_output])
        return model
    
    @staticmethod
    def deep_cnn(input_shape=(128, 128, 3)):
        """Deeper CNN architecture"""
        logger.info("Building Deep CNN architecture")
        
        inputs = keras.Input(shape=input_shape, name='input_image')
        
        # Block 1
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Block 2
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Block 3
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Block 4
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        
        # Age prediction branch
        age_branch = layers.Dense(256, activation='relu', name='age_dense1')(x)
        age_branch = layers.Dropout(0.4)(age_branch)
        age_branch = layers.Dense(128, activation='relu', name='age_dense2')(age_branch)
        age_branch = layers.Dropout(0.3)(age_branch)
        age_output = layers.Dense(1, activation='linear', name='age_output')(age_branch)
        
        # Gender prediction branch
        gender_branch = layers.Dense(128, activation='relu', name='gender_dense1')(x)
        gender_branch = layers.Dropout(0.4)(gender_branch)
        gender_branch = layers.Dense(64, activation='relu', name='gender_dense2')(gender_branch)
        gender_branch = layers.Dropout(0.3)(gender_branch)
        gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')(gender_branch)
        
        model = keras.Model(inputs=inputs, outputs=[age_output, gender_output])
        return model
    
    @staticmethod
    def mobilenet_based(input_shape=(128, 128, 3)):
        """MobileNet-based architecture"""
        logger.info("Building MobileNet-based architecture")
        
        # Load pre-trained MobileNetV2 as base
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        inputs = keras.Input(shape=input_shape, name='input_image')
        
        # Preprocess input for MobileNet
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        
        # Age prediction branch
        age_branch = layers.Dense(128, activation='relu', name='age_dense1')(x)
        age_branch = layers.Dropout(0.3)(age_branch)
        age_output = layers.Dense(1, activation='linear', name='age_output')(age_branch)
        
        # Gender prediction branch
        gender_branch = layers.Dense(64, activation='relu', name='gender_dense1')(x)
        gender_branch = layers.Dropout(0.3)(gender_branch)
        gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')(gender_branch)
        
        model = keras.Model(inputs=inputs, outputs=[age_output, gender_output])
        return model
    
    @staticmethod
    def get_model(architecture_name, input_shape=(128, 128, 3)):
        """Get model by architecture name"""
        architectures = {
            'simple_cnn': ModelArchitectures.simple_cnn,
            'deep_cnn': ModelArchitectures.deep_cnn,
            'mobilenet': ModelArchitectures.mobilenet_based
        }
        
        if architecture_name not in architectures:
            logger.error(f"Unknown architecture: {architecture_name}")
            logger.info(f"Available architectures: {list(architectures.keys())}")
            return None
        
        return architectures[architecture_name](input_shape)