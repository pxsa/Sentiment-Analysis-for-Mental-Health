import re
import pandas as pd
import numpy as np
from typing import List, Tuple
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import torch

# ==================================================
# ==================================================

# # Download required NLTK data
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('stopwords')

# Method 1: Disable SSL verification (temporary fix)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Now download
nltk.download('punkt')
nltk.download('stopwords')



class TextPreprocessor:
    def __init__(self, language='english', remove_stopwords=True, lowercase=True):
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        
        if self.remove_stopwords:
            try:
                self.stop_words = set(stopwords.words(language))
            except:
                print(f"Stopwords for {language} not available, skipping stopword removal")
                self.remove_stopwords = False
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits (keep only letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stop_words(self, text: str) -> str:
        """Remove stopwords from text"""
        if not self.remove_stopwords:
            return text
        
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(filtered_tokens)
    
    def preprocess(self, text: str) -> str:
        """Apply all preprocessing steps"""
        text = self.clean_text(text)
        text = self.remove_stop_words(text)
        return text


class Vocabulary:
    def __init__(self, max_vocab_size=None, min_freq=1):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = {}
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        # Count word frequencies
        for text in texts:
            for word in text.split():
                self.word_freq[word] = self.word_freq.get(word, 0) + 1
        
        # Filter by minimum frequency
        filtered_words = [(word, freq) for word, freq in self.word_freq.items() 
                         if freq >= self.min_freq]
        
        # Sort by frequency
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        
        # Limit vocabulary size
        if self.max_vocab_size:
            filtered_words = filtered_words[:self.max_vocab_size - 2]  # -2 for PAD and UNK
        
        # Build word2idx and idx2word
        for idx, (word, _) in enumerate(filtered_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def encode(self, text: str) -> List[int]:
        """Convert text to indices"""
        return [self.word2idx.get(word, 1) for word in text.split()]
    
    def decode(self, indices: List[int]) -> str:
        """Convert indices to text"""
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices])
    
    def __len__(self):
        return len(self.word2idx)


def pad_sequences(sequences: List[List[int]], max_len: int, padding_value=0) -> np.ndarray:
    """Pad sequences to the same length"""
    padded = np.full((len(sequences), max_len), padding_value, dtype=np.int64)
    
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]
    
    return padded


def prepare_data(df: pd.DataFrame, 
                 text_column: str,
                 label_column: str,
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 max_vocab_size: int = 10000,
                 min_freq: int = 2,
                 max_len: int = 100,
                 language: str = 'english',
                 remove_stopwords: bool = True,
                 random_state: int = 42) -> Tuple:
    """
    Prepare data for training
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, vocab, label_encoder
    """
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(language=language, remove_stopwords=remove_stopwords)
    
    # Preprocess texts
    print("Preprocessing texts...")
    df['processed_text'] = df[text_column].apply(preprocessor.preprocess)
    
    # Remove empty texts
    df = df[df['processed_text'].str.len() > 0].reset_index(drop=True)
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df[label_column])
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        df['processed_text'].values, labels, 
        test_size=test_size, random_state=random_state, stratify=labels
    )
    
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    # Build vocabulary on training data only
    print("Building vocabulary...")
    vocab = Vocabulary(max_vocab_size=max_vocab_size, min_freq=min_freq)
    vocab.build_vocab(X_train)
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Encode texts
    print("Encoding texts...")
    X_train_encoded = [vocab.encode(text) for text in X_train]
    X_val_encoded = [vocab.encode(text) for text in X_val]
    X_test_encoded = [vocab.encode(text) for text in X_test]
    
    # Pad sequences
    print("Padding sequences...")
    X_train_padded = pad_sequences(X_train_encoded, max_len)
    X_val_padded = pad_sequences(X_val_encoded, max_len)
    X_test_padded = pad_sequences(X_test_encoded, max_len)
    
    return (X_train_padded, X_val_padded, X_test_padded,
            y_train, y_val, y_test,
            vocab, label_encoder)


def load_fasttext_embeddings(fasttext_path: str, vocab: Vocabulary, embedding_dim: int = 300):
    """Load FastText embeddings for vocabulary"""
    import fasttext
    
    print(f"Loading FastText model from {fasttext_path}...")
    ft_model = fasttext.load_model(fasttext_path)
    
    # Initialize embedding matrix
    embedding_matrix = np.random.randn(len(vocab), embedding_dim).astype(np.float32) * 0.01
    
    # Set PAD embedding to zeros
    embedding_matrix[0] = np.zeros(embedding_dim)
    
    # Fill embeddings for words in vocabulary
    found = 0
    for word, idx in vocab.word2idx.items():
        if word in ['<PAD>', '<UNK>']:
            continue
        try:
            embedding_matrix[idx] = ft_model.get_word_vector(word)
            found += 1
        except:
            pass
    
    print(f"Found embeddings for {found}/{len(vocab)} words")
    
    return torch.FloatTensor(embedding_matrix)