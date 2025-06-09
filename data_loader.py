import os
import cv2
import numpy as np
import logging
import tensorflow as tf
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class UTKFaceDataLoader:
    def __init__(self, data_dir="data\\utkface", img_size=(128, 128), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.image_paths = []
        self.ages = []
        self.genders = []
        self.train_filenames = []
        self.val_filenames = []
        self.test_filenames = []
    
    def parse_filename(self, filename):
        """Parse UTKFace filename to extract age and gender"""
        try:
            # UTKFace format: [age]_[gender]_[race]_[date&time].jpg
            parts = filename.split('_')
            if len(parts) >= 2:
                age = int(parts[0])
                gender = int(parts[1])  # 0: male, 1: female
                return age, gender
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse filename {filename}: {e}")
        return None, None
    
    def load_image(self, filepath):
        """Load and preprocess image"""
        try:
            image = cv2.imread(filepath)
            if image is None:
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, self.img_size)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            return image
        except Exception as e:
            logger.error(f"Error loading image {filepath}: {e}")
            return None
    
    def load_data(self):
        """Load image paths and labels (not the actual images)"""
        logger.info(f"Loading data from {self.data_dir}")
        
        if not os.path.exists(self.data_dir):
            logger.error(f"Data directory {self.data_dir} does not exist")
            return False
        
        # Look for images in subdirectories
        image_files = []
        subdirs = ['UTKFace', 'crop_part1']
        
        # Check for direct images in the main directory
        direct_images = [f for f in os.listdir(self.data_dir) 
                        if os.path.isfile(os.path.join(self.data_dir, f)) and
                        f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_files.extend([(f, '') for f in direct_images])
        
        # Look in subdirectories
        for subdir in subdirs:
            subdir_path = os.path.join(self.data_dir, subdir)
            if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                subdir_images = [f for f in os.listdir(subdir_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                image_files.extend([(f, subdir) for f in subdir_images])
                logger.info(f"Found {len(subdir_images)} images in {subdir}")
        
        # Check aligned cropped directory which has a nested structure
        aligned_path = os.path.join(self.data_dir, 'utkface_aligned_cropped', 'UTKFace')
        if os.path.exists(aligned_path) and os.path.isdir(aligned_path):
            aligned_images = [f for f in os.listdir(aligned_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_files.extend([(f, os.path.join('utkface_aligned_cropped', 'UTKFace')) for f in aligned_images])
            logger.info(f"Found {len(aligned_images)} images in utkface_aligned_cropped/UTKFace")
        
        if not image_files:
            logger.error("No image files found")
            return False
            
        logger.info(f"Found {len(image_files)} total image files")
        
        valid_count = 0
        for filename, subdir in image_files:
            age, gender = self.parse_filename(filename)
            if age is not None and gender is not None:
                # Construct the full filepath based on subdir
                if subdir:
                    filepath = os.path.join(self.data_dir, subdir, filename)
                else:
                    filepath = os.path.join(self.data_dir, filename)
                
                # Store the path instead of loading the image
                self.image_paths.append(filepath)
                self.ages.append(age)
                self.genders.append(gender)
                valid_count += 1
                
                if valid_count % 1000 == 0:
                    logger.info(f"Processed {valid_count} images...")
        
        logger.info(f"Successfully processed {valid_count} images")
          # Convert to numpy arrays (only for ages and genders - not images)
        self.ages = np.array(self.ages)
        self.genders = np.array(self.genders)
        
        return valid_count > 0
        
    def get_splits(self, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        if len(self.image_paths) == 0:
            logger.error("No data loaded. Call load_data() first.")
            return None
        
        # Check if we have enough samples for stratification
        # Get class counts for genders
        unique_genders, gender_counts = np.unique(self.genders, return_counts=True)
        can_stratify = np.all(gender_counts >= 2)  # Check if all classes have at least 2 samples
        
        logger.info(f"Gender class distribution: {dict(zip(unique_genders, gender_counts))}")
        
        # First split: train+val vs test
        try:
            if can_stratify:
                logger.info("Using stratified split for gender classes")
                paths_temp, paths_test, y_age_temp, y_age_test, y_gender_temp, y_gender_test = \
                    train_test_split(self.image_paths, self.ages, self.genders, 
                                   test_size=test_size, random_state=42, stratify=self.genders)
            else:
                logger.warning("Not enough samples for stratification, using random split")
                paths_temp, paths_test, y_age_temp, y_age_test, y_gender_temp, y_gender_test = \
                    train_test_split(self.image_paths, self.ages, self.genders, 
                                   test_size=test_size, random_state=42)
                                   
            # Second split: train vs val
            val_size_adjusted = val_size / (1 - test_size)
            
            # Check if we have enough samples for stratification in the temp set
            unique_genders_temp, gender_counts_temp = np.unique(y_gender_temp, return_counts=True)
            can_stratify_temp = np.all(gender_counts_temp >= 2)
            
            if can_stratify_temp:
                paths_train, paths_val, y_age_train, y_age_val, y_gender_train, y_gender_val = \
                    train_test_split(paths_temp, y_age_temp, y_gender_temp,
                                   test_size=val_size_adjusted, random_state=42, stratify=y_gender_temp)
            else:
                logger.warning("Not enough samples for validation stratification, using random split")
                paths_train, paths_val, y_age_train, y_age_val, y_gender_train, y_gender_val = \
                    train_test_split(paths_temp, y_age_temp, y_gender_temp,
                                   test_size=val_size_adjusted, random_state=42)
        except ValueError as e:
            logger.error(f"Error during data splitting: {str(e)}")
            logger.warning("Falling back to simple random split without stratification")
            
            # Perform simple random splits as fallback
            paths_temp, paths_test, y_age_temp, y_age_test, y_gender_temp, y_gender_test = \
                train_test_split(self.image_paths, self.ages, self.genders, 
                               test_size=test_size, random_state=42)
            
            val_size_adjusted = val_size / (1 - test_size)
            paths_train, paths_val, y_age_train, y_age_val, y_gender_train, y_gender_val = \
                train_test_split(paths_temp, y_age_temp, y_gender_temp,
                               test_size=val_size_adjusted, random_state=42)
        
        # Store split paths for later use
        self.train_filenames = paths_train
        self.val_filenames = paths_val
        self.test_filenames = paths_test
        
        logger.info(f"Data splits - Train: {len(paths_train)}, Val: {len(paths_val)}, Test: {len(paths_test)}")
        
        return {
            'paths_train': paths_train, 'paths_val': paths_val, 'paths_test': paths_test,
            'y_age_train': y_age_train, 'y_age_val': y_age_val, 'y_age_test': y_age_test,
            'y_gender_train': y_gender_train, 'y_gender_val': y_gender_val, 'y_gender_test': y_gender_test
        }
    def _preprocess_image(self, file_path):
        """Preprocess a single image for TensorFlow dataset"""
        try:
            # Read the image file
            img = tf.io.read_file(file_path)
            # Decode the image
            img = tf.image.decode_jpeg(img, channels=3)
            # Resize the image
            img = tf.image.resize(img, self.img_size)
            # Normalize the image
            img = img / 255.0
            return img
        except Exception as e:
            logger.error(f"Error preprocessing image {file_path}: {e}")
            # Return a blank image in case of error
            return tf.zeros((*self.img_size, 3), dtype=tf.float32)
    
    def _create_dataset(self, file_paths, ages, genders):
        """Create a TensorFlow dataset from file paths and labels"""
        # Create a dataset from the file paths
        path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
        # Map the file paths to preprocessed images
        image_ds = path_ds.map(
            lambda x: self._preprocess_image(x),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Create datasets from the labels
        age_ds = tf.data.Dataset.from_tensor_slices(ages)
        gender_ds = tf.data.Dataset.from_tensor_slices(genders)
        
        # Create a dataset with tuples of (image, (age, gender))
        ds = tf.data.Dataset.zip((
            image_ds, 
            (age_ds, gender_ds)
        ))
        
        # Return batched dataset
        return ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
    def get_tf_datasets(self):
        """Create TensorFlow datasets for training, validation, and testing"""
        if not self.train_filenames:
            logger.info("No data splits found. Creating splits now.")
            splits = self.get_splits()
            if splits is None:
                logger.error("Failed to create data splits.")
                return None, None, None
        else:
            # Use the existing splits already stored in class attributes
            logger.info("Using existing data splits.")
            splits = {
                'paths_train': self.train_filenames,
                'paths_val': self.val_filenames,
                'paths_test': self.test_filenames,
                'y_age_train': self.ages[np.isin(self.image_paths, self.train_filenames)],
                'y_age_val': self.ages[np.isin(self.image_paths, self.val_filenames)],
                'y_age_test': self.ages[np.isin(self.image_paths, self.test_filenames)],
                'y_gender_train': self.genders[np.isin(self.image_paths, self.train_filenames)],
                'y_gender_val': self.genders[np.isin(self.image_paths, self.val_filenames)],
                'y_gender_test': self.genders[np.isin(self.image_paths, self.test_filenames)]
            }
        
        # Create training dataset
        train_ds = self._create_dataset(
            splits['paths_train'],
            splits['y_age_train'],
            splits['y_gender_train']
        )
        
        # Create validation dataset
        val_ds = self._create_dataset(
            splits['paths_val'],
            splits['y_age_val'],
            splits['y_gender_val']
        )
        
        # Create test dataset
        test_ds = self._create_dataset(
            splits['paths_test'],
            splits['y_age_test'],
            splits['y_gender_test']
        )
        
        logger.info("TensorFlow datasets created successfully")
        
        return train_ds, val_ds, test_ds