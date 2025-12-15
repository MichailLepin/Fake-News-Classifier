# Presentation Notes: Fake News Classification System

## Slide 1: Title
> "Hello everyone. Today I'm presenting our Fake News Classification System — an NLP-based solution for detecting misinformation using deep learning."

---

## Slide 2: Project Overview
> "Our goal was to build an automated fake news detection system with multiple models for comparison and a production-ready web interface.

> The target audience includes journalists, fact-checkers, and content moderators who need to quickly verify news authenticity.

> We used Python with FastAPI for backend, PyTorch for machine learning, and deployed on Railway with GitHub Pages for the frontend."

---

## Slide 3: Datasets
> "We worked with two datasets. The ISOT Kaggle dataset contains about 45,000 news articles, roughly balanced between fake and real news.

> The LIAR dataset has nearly 13,000 political statements. We converted its 6-level labels to binary classification.

> Data was split 64-16-20 for training, validation, and testing with stratification to maintain class balance."

---

## Slide 4: Model Architectures
> "We implemented four models. Two baseline models: CNN with convolutional filters for n-gram patterns, and bidirectional LSTM for capturing long-range dependencies. Both use GloVe embeddings.

> For transformer models, we fine-tuned BERT-base-uncased with 12 layers for highest accuracy, and DistilBERT which is 40% smaller but nearly as accurate."

---

## Slide 5: Training Configuration
> "Training used batch size 16, learning rate 2e-5, with early stopping after 3 epochs without improvement.

> We used Adam optimizer for baseline models and AdamW with warmup scheduler for transformers.

> All training was done in Google Colab with GPU acceleration. F1-score was our primary evaluation metric."

---

## Slide 6: Model Performance
> "Here are the expected results. Transformer models achieve 93-96% accuracy, while baseline models reach 85-93%.

> CNN offers the fastest inference at about 38 milliseconds, while BERT takes around 120ms.

> DistilBERT provides a good balance — nearly BERT-level accuracy at half the inference time."

---

## Slide 7: Web Interface
> "The web interface has two main sections. The EDA Dashboard shows dataset statistics: label distributions, text lengths, and top words for each class.

> The Model Demo allows users to input text, select a model, and get real-time predictions with confidence scores and attention visualization showing which words influenced the prediction."

---

## Slide 8: System Architecture
> "The architecture separates frontend and backend. The frontend on GitHub Pages sends REST API requests to the FastAPI backend on Railway.

> The backend loads models on startup, processes text, and returns predictions. If the API is unavailable, the frontend gracefully falls back to a simulation mode."

---

## Slide 9: Deployment
> "For deployment, Railway uses NIXPACKS to build and run our Python application with Uvicorn.

> Model files include PyTorch weights for CNN/LSTM and full model directories for BERT/DistilBERT.

> We created utility scripts for deployment preparation and model downloading, plus comprehensive documentation."

---

## Slide 10: Conclusion & Future Work
> "In summary, we successfully trained four models, built a full web interface with model comparison and attention visualization, and deployed everything to production.

> Future improvements could include real transformer attention weights, ensemble methods, user history, and batch processing capabilities.

> Thank you for your attention. The project is available on GitHub."

---

## Timing Guide
- **Total presentation:** ~8-10 minutes
- **Per slide:** ~45-60 seconds
- **Leave 2-3 minutes for questions**

