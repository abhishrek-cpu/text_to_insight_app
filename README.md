# 🧠 Text to Insight – AI-Powered Text Analysis Web App

This Streamlit web app extracts insights from any text or document (PDF, Word, or PowerPoint).  
It performs **keyword extraction**, **sentiment analysis**, and **auto summarization** using modern NLP models.

---

## 🚀 Features

✅ Upload files in **PDF**, **DOCX**, or **PPTX** format  
✅ Extract text automatically from uploaded files  
✅ Perform **keyword extraction** using TF-IDF  
✅ Run **sentiment analysis** with a pretrained transformer model  
✅ Generate concise **summaries** using `DistilBART`  
✅ Clean, user-friendly **Streamlit interface**

---

## Library Stack

- Python 🐍
- Streamlit 🌐
- Transformers (Hugging Face) 🤗
- scikit-learn
- PyMuPDF, python-docx, python-pptx
- NLTK

---

## Installation

```bash
# Clone this repository
git clone https://github.com/abhishrek-cpu/text-to-insight.git
cd text-to-insight

# Create and activate virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate   # (Windows)
# or source venv/bin/activate  (Mac/Linux)

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 🖼️ App Screenshot

![Text to Insight App]<img width="1307" height="713" alt="Image" src="https://github.com/user-attachments/assets/d2194620-24c4-4dc5-81e7-82a2e46f534c" />

## 🔍 Example Output

Here’s an example of summary from an uploaded file:

![Example Output]<img width="1309" height="681" alt="Image" src="https://github.com/user-attachments/assets/6d0d7b51-4c4d-422a-9e93-843994bc92ec" />
