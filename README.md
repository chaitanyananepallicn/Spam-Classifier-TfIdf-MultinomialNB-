# Spam/Ham Classifier (TF-IDF)

A simple but effective web application built with Python and Streamlit that uses a **Term Frequency-Inverse Document Frequency (TF-IDF)** model and a Multinomial Naive Bayes classifier to determine if a message is spam or ham.

***

## Live Application

**You can access a live deployed demo here:**
[**➡️ Live Spam Classifier App**](https://spam-classifier-tfidf-multinomialnb.onrender.com)

***

## 🚀 Features

* **Interactive UI**: A clean and simple web interface built with Streamlit for easy interaction.
* **Real-time Prediction**: Instantly classifies any user-provided message as either Spam or Ham.
* **Confidence Score**: Displays the model's confidence in its prediction with a progress bar.
* **Efficient Backend**: Uses pre-trained Scikit-learn models saved as pickle files for fast loading and prediction.

***

## 🛠️ Tech Stack

* **Core Language**: Python
* **Web Framework**: Streamlit
* **ML Libraries**:
    * `Scikit-learn` for machine learning (TfidfVectorizer, MultinomialNB).
    * `NLTK` for natural language processing (stopwords, lemmatization).

***

## ⚙️ Getting Started

To run this project on your local machine, follow these steps.

### Prerequisites

Make sure you have Python 3.8+ and pip installed on your system.

### Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd <repository-folder>
    ```

2.  **Create and Activate a Virtual Environment**
    * **On macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * **On Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Create a `requirements.txt` file**
    * Create a file named `requirements.txt` and add the following lines:
        ```text
        streamlit
        scikit-learn
        nltk
        ```

4.  **Install the Required Libraries**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Ensure Model Files are Present**
    * Make sure your trained model files, `TfIdf.pickle` and `model.pickle`, are in the root directory of the project.

6.  **Run the Streamlit Application**
    ```bash
    streamlit run app.py
    ```
    * After running the command, open your web browser and navigate to the local URL provided (usually `http://localhost:8501`).

***

## 📂 Project Structure

```text
.
├── app.py              # The main Streamlit application script
├── TfIdf.pickle        # The saved TfidfVectorizer object
├── model.pickle        # The saved MultinomialNB model object
├── requirements.txt    # Python dependencies
└── README.md           # This file
