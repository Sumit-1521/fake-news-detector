# 📰 Fake News Detector

An NLP-based machine learning app that classifies news articles as **Real** or **Fake** using TF-IDF and Random Forest.

---

## 📊 Dataset Used

📌 [Fake and Real News Dataset – Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
- `true.csv` → Real news  
- `fake.csv` → Fake news  
- Combined & labeled into `combined_news.csv`

---

## ⚙️ How to Run

```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
streamlit run app.py   # or python app.py if Flask
