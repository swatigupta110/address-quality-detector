# Address Quality Detector
An NLP-based machine learning project trained on Indian addresses to detect invalid or low-quality addresses. Built to enhance delivery success rates, reduce operational returns, and improve overall data quality in e-commerce and logistics systems.

## Live Demo
https://huggingface.co/spaces/swatigupta110/space-address-quality-detector

## Detailed Explanation of Project 
This project builds a machine learning pipeline to automatically evaluate the quality of addresses using **Natural Language Processing** techniques.

### Data Preparation
- Address data was extracted from datbase using **SQL query** and csv file loaded using **Pandas**.
- Missing values and duplicate records were removed to ensure data quality.
- Multiple address components such as ***address, address_line1, landmark, city,*** and ***state*** were combined into a single text feature for model training.

### Text Preprocessing
Several NLP preprocessing techniques were applied:
- **Tokenization** using NLTK [RegexpTokenizer](https://codefinity.com/courses/v2/c68c1f2e-2c90-4d5d-8db9-1e97ca89d15e/72edaf8c-5c21-464c-8076-a55d95ee356d/c19fdf12-75ce-4032-b0a0-d9ff9cd9e002)
- **Stopword removal** using NLTK English stopword list
- Number **normalization**, replacing numeric patterns with a `<number>` token
- **Stemming** using the [Lancaster Stemmer](https://www.baeldung.com/cs/porter-vs-lancaster-stemming-algorithms)
- Cleaning extra spaces and formatting the text
These steps help normalize address patterns and improve model learning.

### Train-Test Split
The dataset was divided into:
- 80% Training Data
- 20% Testing Data
A **stratified split** was used to maintain equal distribution of address quality labels across training and testing datasets.

### FastText Model Training
A **FastText supervised classification model** was trained using the processed text data.
Key parameters used:
- epoch = 10
- learning rate = 0.3
- wordNgrams = 3
- vector dimension = 10
- character n-grams = 3 to 6
FastText was chosen because it is efficient for text classification and works well with short text patterns such as addresses.

### Model Evaluation
The trained model was evaluated using several metrics:
- Precision
- Recall
- F1 Score
- Confusion Matrix
A confusion matrix heatmap was also generated using Seaborn and Matplotlib to visualize prediction performance.

### Prediction Pipeline
A reusable prediction function was implemented to:
- Apply the same preprocessing steps to new address inputs
- Load the trained FastText model
- Predict address quality
- Return the predicted label and probability score

### Model Optimization
To improve model efficiency, FastText quantization was applied:
- Reduced model size
- Maintained prediction performance
- Improved deployment efficiency

### Deployment
The trained model was integrated into a Gradio-based web interface and deployed on Hugging Face Spaces, allowing users to test address quality predictions in real time.
