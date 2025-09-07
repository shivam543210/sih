import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@keras.utils.register_keras_serializable()
class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = keras.layers.Embedding(maxlen, embed_dim)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

class DNATransformerService:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model = None
        self.vocab = None
        self.K = None
        self.MAXLEN = None
        self.VOCAB_SIZE = None
        self.idx_to_species = {}
        self.species_to_idx = {}
        self._initialize()

    def _initialize(self):
        """Initialize the model and configurations"""
        try:
            # Load configurations
            self._load_configs()
            
            # Build vocabulary
            self._build_vocab()
            
            # Load the model
            self._load_model()
            
            logger.info(f"✅ Model initialized successfully")
            logger.info(f"📊 Classes: {len(self.idx_to_species)}, K-mer: {self.K}, MaxLen: {self.MAXLEN}")
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {str(e)}")
            raise

    def _load_configs(self):
        """Load vocabulary and label configurations"""
        # Load vocab config
        vocab_path = os.path.join(self.model_dir, "vocab2_config.json")
        with open(vocab_path) as f:
            vocab_cfg = json.load(f)
        
        self.K = vocab_cfg["k"]
        self.MAXLEN = vocab_cfg["maxlen"]
        self.VOCAB_SIZE = vocab_cfg["vocab_size"]
        
        # Load label encoder
        label_path = os.path.join(self.model_dir, "label2_encoder.json")
        with open(label_path) as f:
            label_cfg = json.load(f)
        
        self.idx_to_species = {i: name for i, name in enumerate(label_cfg["classes"])}
        self.species_to_idx = {v: k for k, v in self.idx_to_species.items()}

    def _build_vocab(self):
        """Build k-mer vocabulary"""
        DNA = "ACGT"
        
        def generate_kmers(prefix, k):
            if k == 0:
                return [prefix]
            result = []
            for base in DNA:
                result.extend(generate_kmers(prefix + base, k-1))
            return result
        
        all_kmers = generate_kmers("", self.K)
        self.vocab = {kmer: idx+1 for idx, kmer in enumerate(all_kmers)}

    def _load_model(self):
        """Load the trained transformer model"""
        model_path = os.path.join(self.model_dir, "khushi2_transformer.keras")
        
        self.model = keras.models.load_model(
            model_path,
            custom_objects={"TokenAndPositionEmbedding": TokenAndPositionEmbedding}
        )

    def _extract_kmers(self, sequence: str) -> List[str]:
        """Extract k-mers from DNA sequence"""
        DNA = "ACGT"
        seq = sequence.upper()
        return [seq[i:i+self.K] for i in range(len(seq) - self.K + 1) 
                if set(seq[i:i+self.K]).issubset(set(DNA))]

    def _encode_sequence(self, sequence: str) -> np.ndarray:
        """Encode DNA sequence to k-mer token IDs"""
        # Clean sequence - keep only ACGT
        DNA = "ACGT"
        clean_seq = "".join([c for c in sequence.upper() if c in DNA])
        
        if len(clean_seq) < self.K:
            raise ValueError(f"Sequence too short after cleaning. Minimum: {self.K} nucleotides")
        
        # Extract k-mers
        kmers = self._extract_kmers(clean_seq)
        
        if not kmers:
            raise ValueError("No valid k-mers found in sequence")
        
        # Convert to token IDs
        token_ids = [self.vocab.get(kmer, 0) for kmer in kmers]
        
        # Pad sequence
        padded = keras.preprocessing.sequence.pad_sequences(
            [token_ids], 
            maxlen=self.MAXLEN, 
            padding="post", 
            truncating="post"
        )[0]
        
        return padded

    def predict(self, sequence: str) -> Dict:
        """Make prediction for a single DNA sequence"""
        try:
            # Validate input
            if not sequence or len(sequence.strip()) == 0:
                raise ValueError("Empty sequence provided")
            
            # Encode sequence
            encoded_seq = self._encode_sequence(sequence)
            x = np.expand_dims(encoded_seq, axis=0)
            
            # Make prediction
            predictions = self.model.predict(x, verbose=0)
            
            # Get results
            pred_class = int(np.argmax(predictions, axis=1)[0])
            confidence = float(predictions[0, pred_class])
            species = self.idx_to_species[pred_class]
            
            # Get all class probabilities
            all_probabilities = {
                self.idx_to_species[i]: float(prob)
                for i, prob in enumerate(predictions[0])
            }
            
            return {
                "species": species,
                "confidence": confidence,
                "probabilities": all_probabilities,
                "model_info": {
                    "k_mer_size": self.K,
                    "sequence_length": len(sequence),
                    "processed_length": len("".join([c for c in sequence.upper() if c in "ACGT"])),
                    "num_kmers": len(self._extract_kmers(sequence))
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def batch_predict(self, sequences: List[str]) -> List[Dict]:
        """Make predictions for multiple sequences"""
        results = []
        for i, seq in enumerate(sequences):
            try:
                result = self.predict(seq)
                result["sequence_id"] = i
                results.append(result)
            except Exception as e:
                results.append({
                    "sequence_id": i,
                    "error": str(e),
                    "species": None,
                    "confidence": 0.0
                })
        return results

    def health_check(self) -> Dict:
        """Check if model is loaded and ready"""
        return {
            "model_loaded": self.model is not None,
            "vocab_ready": self.vocab is not None,
            "num_classes": len(self.idx_to_species),
            "supported_species": list(self.idx_to_species.values()),
            "k_mer_size": self.K,
            "max_sequence_length": self.MAXLEN
        }
