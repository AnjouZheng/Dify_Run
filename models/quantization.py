import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

class SpamModelQuantizer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        try:
            model = torch.load(self.model_path)
            self.logger.info(f"Successfully loaded model from {self.model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            raise

    def quantize_model(self, model, quantization_type='dynamic'):
        try:
            if quantization_type == 'dynamic':
                quantized_model = quantize_dynamic(
                    model, 
                    {nn.Linear, nn.LSTM}, 
                    dtype=torch.qint8
                )
                self.logger.info("Model quantized using dynamic quantization")
            else:
                raise ValueError("Unsupported quantization type")
            
            return quantized_model
        except Exception as e:
            self.logger.error(f"Quantization failed: {e}")
            raise

    def save_quantized_model(self, quantized_model, output_path: str):
        try:
            torch.save(quantized_model.state_dict(), output_path)
            self.logger.info(f"Quantized model saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")
            raise

    def quantize_and_save(self, output_path: str):
        model = self.load_model()
        quantized_model = self.quantize_model(model)
        self.save_quantized_model(quantized_model, output_path)

def main():
    quantizer = SpamModelQuantizer('/models/trained_models/spam_classifier.pth')
    quantizer.quantize_and_save('/models/trained_models/spam_classifier_quantized.pth')

if __name__ == '__main__':
    main()