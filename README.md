## Week 1: Python & Core Libraries
- Gained hands-on experience with Python syntax and basic programming constructs.
- Learned NumPy for numerical computations, array operations, broadcasting, etc.
- Explored Pandas for data manipulation and analysis using Series and DataFrames.
- Used Matplotlib for visualizing data through line plots, bar charts, histograms, etc.
- Got introduced to PyTorch and its Tensor operations for deep learning workflows.

## Week 2: Natural Language Processing (NLP)
- Studied basic NLP concepts such as text cleaning, tokenization, and stopword removal.
- Practiced Regular Expressions (regex) for pattern-based text extraction.
- Explored word embeddings like Word2Vec and GloVe for representing text numerically.
- Performed basic sentiment analysis to classify text as positive or negative.
- Compared NLP libraries: NLTK (granular and academic) vs spaCy (production-ready and efficient).

## Week 3-4: Neural Networks & Optimizers
- Learned the structure and working of artificial neural networks (ANNs).
- Built and understood feedforward neural networks with activation functions like ReLU and Sigmoid.
- Studied Recurrent Neural Networks (RNNs) for sequential data processing.
- Explored optimizers such as SGD, Adam, and RMSprop for efficient training.
- Understood concepts of overfitting, regularization (like Dropout), and early stopping.

## Week 5-6: LLMs, RAG, LangChain
- Gained a solid understanding of Transformer architectures and Large Language Models (LLMs).
- Explored decoding strategies and fine-tuning techniques for LLMs using Hugging Face.
- Implemented Retrieval-Augmented Generation (RAG) using LangChain and vector databases.
- Built a question-answering chatbot powered by LLMs and PDF-based context retrieval.
- Designed a local user interface using Gradio for seamless interaction with the model.

---

## Assignment 1: Digit Recognizer with PyTorch

### Objective
To build a digit classification model using PyTorch that predicts handwritten digits from the MNIST dataset.

### Tasks Completed
- **Data Loading:** Used `torchvision.datasets.MNIST` and `DataLoader` to prepare training and testing sets.
- **Model Definition:** Implemented a simple feedforward neural network with multiple linear layers and ReLU activations.
- **Training:** Trained using `CrossEntropyLoss` and the `Adam` optimize

## Assignment 2: Sentiment Analysis with LSTM using PyTorch 
- Gained hands-on experience with text preprocessing techniques such as tokenization, padding, and vocabulary indexing for NLP tasks.
- Built and trained an LSTM-based sentiment classification model using PyTorch from scratch.
- Explored dataset distribution, tackled class imbalance (if any), and evaluated performance using metrics like accuracy, precision, and recall.
- Experimented with various hyperparameters including embedding size, LSTM hidden units, and learning rate to optimize model performance.
- Visualized model performance (e.g., training curves or confusion matrix) and documented the entire ML pipeline from data loading to prediction.

## Final Project: Chatbot over PDF using RAG
- Understood how to load, parse, and chunk large PDF documents using tools like PyMuPDF and LangChain's text splitter.
- Learned how to implement Retrieval-Augmented Generation (RAG) by combining a vector database (FAISS) with LLM inference for context-aware responses.
- Integrated the LLaMA 2 (7B) model from Hugging Face and built a query-response pipeline using Transformers and Accelerate for efficient memory usage.
- Explored vector embeddings with sentence-transformers to encode PDF chunks and retrieve top relevant context for a given user query.
- Built an interactive Gradio UI for querying the chatbot and tested it using the UG Rulebook PDF.

