# SPEECH-EMOTION-DETECTION



---

## 🎯 Objectives

- To detect human emotions from speech audio using machine learning and deep learning.
- To extract relevant features (MFCC) for traditional model training.
- To explore transformer-based models like wav2vec2 for end-to-end feature learning.
- To compare the performance of LSTM vs wav2vec2-based models.

---

## 📚 Datasets Used

The project uses publicly available emotional speech datasets:

1. [TESS – Toronto Emotional Speech Set](https://tspace.library.utoronto.ca/handle/1807/24487)
2. [RAVDESS – Ryerson Audio-Visual Database of Emotional Speech and Song](https://zenodo.org/record/1188976)
3. [SAVEE – Surrey Audio-Visual Expressed Emotion Dataset](https://kahlan.eps.surrey.ac.uk/savee/)
4. [CREMA-D – Crowd-Sourced Emotional Multimodal Actors Dataset](https://zenodo.org/record/3819440)

These datasets include recordings of actors expressing different emotions like happiness, anger, sadness, fear, disgust, surprise, and neutral.

---

## 🔬 Methodology

### Step 1: **Data Preprocessing**
- Audio files are loaded in `.wav` format.
- Sample rates are standardized (commonly to 16 kHz).
- Silence trimming and noise reduction (if necessary).
- Labels are extracted from file names or CSV metadata.

### Step 2: **Feature Extraction**
- **MFCC (Mel Frequency Cepstral Coefficients)** are extracted using `librosa`.
  - These capture the timbral and pitch characteristics of human voice.
  - Typically, 13–40 MFCCs are computed per frame.
- The MFCC features are padded or truncated to a fixed length for uniform input dimensions.

### Step 3: **Modeling Approaches**
#### A. LSTM-Based Model:
- A Sequential LSTM model is built using TensorFlow/Keras.
- Input: MFCC features (2D array of frames × coefficients).
- LSTM layers learn temporal patterns in speech.
- Dense output layer with softmax activation classifies the emotion.

#### B. Transformer-Based Model (wav2vec2 - optional):
- Pretrained wav2vec2 model (from HuggingFace Transformers) is used.
- Audio is fed directly to the model without MFCC extraction.
- wav2vec2 encodes audio into contextual embeddings.
- A classification head is added and fine-tuned for emotion classification.

### Step 4: **Model Training**
- Train-test split is applied (e.g., 80–20).
- Data is shuffled and normalized.
- Categorical crossentropy loss and Adam optimizer are used.
- Models are trained with callbacks like EarlyStopping and ModelCheckpoint.

### Step 5: **Evaluation**
- Performance is evaluated using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
- Experiments are conducted across different datasets to test generalizability.

---

## 🧠 Models Used

### ✅ MFCC + LSTM
- Easy to implement
- Performs well on structured and clean data
- Learns temporal dependencies

### ✅ wav2vec2 (optional)
- State-of-the-art transformer for audio
- Learns features directly from raw waveform
- Requires more computation but performs well even with less preprocessing

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/speech-emotion-detection.git
cd speech-emotion-detection
