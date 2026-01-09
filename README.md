🧠 Assignment 3 – NLP Clothing Reviews Web Application
📘 Description

This project applies Natural Language Processing (NLP) techniques to analyze clothing reviews and display them through a Flask web interface.
Users can browse clothing items, view descriptions and ratings, add new reviews, and search for products based on similarity using TF-IDF.
The application integrates text preprocessing, feature representation, and basic recommendation filtering.

📂 Project Structure
File / Folder	Description
assignment3.csv	Dataset containing clothing reviews and product metadata
stopwords_en.txt	English stopword list used in preprocessing
templates/	HTML templates for Flask (base.html, index.html, details.html, etc.)
static/	CSS, JS, and image files
app.py	Main Flask application
requirements.txt	Python dependencies
vocab.txt	Alphabetically sorted unigram vocabulary (generated after preprocessing)
count_vectors.txt	Count vector representations (generated after preprocessing)
🧹 Text Preprocessing Steps

Convert all text to lowercase

Remove punctuation, digits, and special symbols using regex

Tokenize the text into individual words

Remove stopwords from stopwords_en.txt

Generate vocab.txt and count_vectors.txt

Optionally, build weighted TF-IDF or embedding-based document vectors

🚀 How to Run the Web Application
1️⃣ Create & Activate a Virtual Environment (Recommended)

Using Conda:

conda create -n projectFlask python=3.12
conda activate projectFlask


Or using venv:

python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Start the Flask Server
python app.py


If successful, you’ll see output similar to:

* Running on http://127.0.0.1:5000/

4️⃣ Open the Webpage

Visit http://127.0.0.1:5000/
 in your browser.

🏠 Homepage: Displays clothing items and descriptions

🔍 Search Bar: Find products by keyword similarity (TF-IDF)

💬 Details Page: Read and add reviews for individual items

📤 Outputs (Generated During Tasks)

vocab.txt → Alphabetically sorted unigram vocabulary

count_vectors.txt → Bag-of-Words representation of reviews

weighted_vectors.txt (optional) → TF-IDF or embedding-based representations

👨‍💻 Author

Hishikesh Phukan (s4031214)
Master of Data Science, RMIT University

📅 October 2025