import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_file="model_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    def _load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return {"models": [], "current_version": 0}
        return {"models": [], "current_version": 0}
    def save_config(self):
        """Save configuration to file"""
        try:
            # Create a copy of the config to modify for serialization
            config_to_save = self._prepare_config_for_json(self.config)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=4, ensure_ascii=False)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def _prepare_config_for_json(self, obj):
        """Recursively prepare objects for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._prepare_config_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_config_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to lists
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    def add_model(self, model_params, model_path, training_results):
        """Add new model configuration"""
        import copy
        version = self.config["current_version"] + 1
        
        # Make a deep copy of results to avoid modifying the original data
        # This is important as training_results might be used elsewhere
        results_copy = copy.deepcopy(training_results)
        
        # Remove large arrays from results to prevent excessive JSON size
        if 'predictions' in results_copy:
            # Store summary statistics about predictions instead of full arrays
            age_pred_len = len(results_copy['predictions']['age_pred']) if 'age_pred' in results_copy['predictions'] else 0
            gender_pred_len = len(results_copy['predictions']['gender_pred']) if 'gender_pred' in results_copy['predictions'] else 0
            
            # Replace full prediction arrays with just their sizes
            results_copy['predictions'] = {
                'age_predictions_count': age_pred_len,
                'gender_predictions_count': gender_pred_len
            }
        
        model_config = {
            "version": version,
            "model_path": model_path,
            "created_at": datetime.now().isoformat(),
            "parameters": model_params,
            "training_results": results_copy,
            "status": "active"
        }
        
        self.config["models"].append(model_config)
        self.config["current_version"] = version
        self.save_config()
        
        logger.info(f"Added model version {version} to configuration")
        return version
    
    def get_available_models(self):
        """Get list of available models"""
        return [(m["version"], m["created_at"], m["parameters"]["architecture"]) 
                for m in self.config["models"] if m["status"] == "active"]
    
    def get_model_path(self, version):
        """Get model path by version"""
        for model in self.config["models"]:
            if model["version"] == version:
                return model["model_path"]
        return None
    
    def get_latest_model(self):
        """Get latest model information"""
        if self.config["models"]:
            return self.config["models"][-1]
        return None