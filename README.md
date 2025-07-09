# nlp-drug-side-effect-pred-machkevds

# Adverse Drug Event (ADE) Classification: AI for Drug Safety

An NLP project leveraging **Transformer models (BERT)** to automatically classify sentences describing **Adverse Drug Events (ADEs)** in healthcare text. This showcases a full ML workflow, from data analysis to advanced model evaluation, providing a robust solution for enhancing drug safety monitoring.

---

## Key Features & Project Impact

* **Precision ADE Detection:** Model classifies sentences as either **"Related"** (contains an ADE) or **"Not-Related"** (does not contain an ADE).
* **Significant Performance Improvement:** Achieved a **macro F1-score of 0.94** on the test set, with a critical **0.92 F1-score for the 'Related' (ADE) class**. This is a substantial leap from the traditional ML baseline's 0.78 F1-score for ADEs.
* **Reduced Critical Errors:** The Transformer model drastically reduced **False Negatives** (missed ADEs) by 62% (from 260 to 99) and **False Positives** (false alarms) by 63% (from 527 to 193) compared to the baseline.

---

## Technical Highlights

* **Dataset:** `ADE-Corpus-V2` from Hugging Face Hub (17,637 training, 5,879 test sentences), identified as having a significant **~2.5:1 class imbalance** (Not-Related:Related).
* **Preprocessing:** Basic text cleaning (lowercasing, punctuation removal) and Transformer-specific tokenization (subword units, `[CLS]`, `[SEP]`, attention masks, padding).
* **Baseline Model:** Logistic Regression trained on Bag-of-Words features, establishing a F1-score baseline of **0.78 for the 'Related' class**.
* **Advanced Model:** Fine-tuned **`bert-base-uncased` (110M parameters)** using the Hugging Face `transformers` library and `Trainer` API.
* **Evaluation Metrics:** Comprehensive use of Precision, Recall, F1-score, and Confusion Matrices for both overall and per-class performance.
* **Model Persistence:** Implemented saving and loading of the 417 MB trained model and tokenizer to Google Drive, ensuring reusability without re-training. The model is also ready for upload to Hugging Face Hub.

---

## Next Steps

This project successfully demonstrates the technical capabilities of building a high-performing NLP model for a critical healthcare task. While the core implementation is complete, the journey from development to real-world impact involves further considerations around deployment strategies, continuous monitoring, and the broader implications of AI in sensitive domains. These aspects are crucial for moving from a successful proof-of-concept to a production-ready solution.

---

## Key Learnings & Challenges

* **Full ML Lifecycle Mastery:** Gained hands-on experience across the entire ML project pipeline: data acquisition, rigorous EDA, traditional ML baselines, advanced deep learning model training, and in-depth evaluation.
* **Navigating Class Imbalance:** Effectively managed and accounted for dataset imbalance using appropriate strategies and evaluation metrics.
* **Critical Error Analysis:** Performed detailed error analysis, including pinpointing and dissecting a specific instance where the model (despite high overall performance) made a **highly confident False Negative** prediction for a clear ADE sentence.
* **Practical MLOps Foundations:** Learned essential practices for model persistence, understanding of deployment modalities, and the crucial need for post-deployment monitoring.
* **Library Version Management:** Successfully navigated and resolved multiple dependency and version conflicts within the `transformers` ecosystem.

---

## How to Run This Project

To run this project and explore the code:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Open in Google Colab:** Go to `colab.research.google.com`, then `File > Open notebook > GitHub`, and paste your repository URL.
3.  **Set Up Runtime:** Change Colab runtime to `GPU` (`Runtime > Change runtime type`).
4.  **Mount Google Drive:** (If loading the model from Drive)
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
5.  **Install Libraries:**
    ```bash
    !pip install -r requirements.txt -q
    # IMPORTANT: If you hit dependency errors, restart runtime AFTER install, then remount Drive and reinstall.
    ```
6.  **Run the Notebook:**
    * **Option A: Re-train Model:** Run all cells sequentially (approx. 1.5 hours).
    * **Option B: Load Pre-trained Model:** Update the `load_directory` variable to your Google Drive path (e.g., `"/content/drive/MyDrive/ml_models/my_ade_bert_model"`) or use the Hugging Face Hub path (e.g., `"your-huggingface-username/your-repo-name-on-hub"`) and run the model loading and evaluation cells.

*Note: The trained model (417MB) is not directly included in this repository to keep it lightweight. Please use the instructions above to load it from Google Drive or Hugging Face Hub.*

---

## Technologies Used

Python, PyTorch, Hugging Face (transformers, datasets, accelerate, evaluate), scikit-learn, pandas, matplotlib, seaborn, Google Colaboratory, Google Drive.

---
