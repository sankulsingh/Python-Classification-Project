# Research Abstract Classification Using Machine Learning

## 1. Project Overview

This project focuses on classifying research-paper abstracts into three academic categories: **Science, Business, and Social Science** using machine learning and natural language processing (NLP) techniques.

The project uses text preprocessing and TF-IDF feature extraction to convert research abstracts into numerical features. Six machine learning classifiers are trained and evaluated to compare their performance on a multiclass text-classification task.

The project also includes an interactive demo in the main Jupyter Notebook, allowing users to enter or upload research-paper data and obtain predicted academic categories.

## 2. Problem Statement

Research papers are published across a wide range of academic disciplines. Manually identifying the academic category of a research paper from its abstract can be time-consuming.

The objective of this project is to develop a machine learning system that automatically classifies research-paper abstracts into one of the following categories:

- **Science**
- **Business**
- **Social Science**

The performance of different machine learning algorithms is compared using standard classification evaluation metrics to identify the most suitable model for this task.

## 3. Dataset Description

The dataset contains approximately 6,000 research-paper abstracts collected from OpenAlex and categorized into three academic fields: Science, Business, and Social Science.

| Property | Description |
|---|---|
| Dataset file | `research_abstracts_6000_cleaned.csv` |
| Data source | OpenAlex |
| Data type | Research-paper abstracts |
| Input feature | `abstract` |
| Target variable | `label` |
| Target categories | Science, Business, Social Science |
| Dataset size | Approximately 6,000 records |
| Duplicate handling | 28 duplicated abstracts were replaced with new abstracts |
| Classification type | Multiclass supervised text classification |

The dataset was prepared to provide research abstracts suitable for training and evaluating supervised machine learning classifiers. During dataset finalization, 28 duplicated abstracts were replaced with new abstracts to maintain the intended dataset size while reducing duplication.

The `abstract` column contains the text used for classification, while the `label` column contains the corresponding academic category.

The finalized CSV dataset is included in the project folder so that the classification notebook can be run without repeating the data collection process.

## 4. Technologies and Libraries Used

The project uses Python and the following libraries:

- **Pandas:** Dataset loading and manipulation.
- **NumPy:** Numerical operations.
- **Scikit-learn:** Text vectorization, model training, and evaluation.
- **Matplotlib:** Visualization of model performance.
- **Seaborn:** Confusion-matrix visualizations.
- **Joblib:** Saving trained models and the TF-IDF vectorizer.
- **Jupyter Notebook / Google Colab:** Development and execution environment.

## 5. Data Preprocessing and Feature Extraction

Text preprocessing and feature extraction are important steps because machine learning algorithms require numerical input rather than raw text.

### TF-IDF Vectorization

The project uses **Term Frequency–Inverse Document Frequency (TF-IDF)** to transform research-paper abstracts into numerical feature vectors.

The vectorizer is configured with the following settings:

- Lowercase conversion enabled.
- Unicode accent stripping.
- Unigrams and bigrams using `ngram_range=(1, 2)`.
- `min_df=2` to ignore terms appearing in fewer than two documents.
- `max_df=0.95` to ignore terms appearing in more than 95% of documents.
- `max_features=50000` to limit the vocabulary size.
- `sublinear_tf=True` to apply sublinear term-frequency scaling.

Using both unigrams and bigrams helps the model capture individual words as well as meaningful two-word expressions.

### Training and Testing Split

The dataset is divided into training and testing sets using an **80:20 split**. Stratification is used to preserve the relative distribution of the target classes in both sets.

The training set is used to train the models, while the testing set is used to evaluate their performance on unseen data.

## 6. Machine Learning Algorithms

Six machine learning classifiers are implemented and compared.

### 6.1 Logistic Regression

Logistic Regression is a linear classification algorithm commonly used for text classification. It estimates the probability that an abstract belongs to each category.

### 6.2 Decision Tree

Decision Tree classifies data by learning a sequence of decision rules. It is easy to interpret but can be prone to overfitting on high-dimensional text features.

### 6.3 Naive Bayes

Multinomial Naive Bayes is a probabilistic algorithm frequently used for document classification. It applies Bayes' theorem with a conditional-independence assumption between features.

### 6.4 K-Nearest Neighbors (kNN)

K-Nearest Neighbors predicts the class of a document based on the classes of its nearest training examples. In this project, it uses five neighbors and cosine distance.

### 6.5 Support Vector Machine (SVM)

The project uses a linear Support Vector Machine implemented through `LinearSVC`. It learns decision boundaries that separate the categories in the TF-IDF feature space.

### 6.6 Random Forest

Random Forest combines multiple decision trees to produce a classification result. It can model complex relationships but may be less effective than linear models for sparse, high-dimensional text features.

## 7. Model Evaluation

The classifiers are evaluated using the following metrics:

- **Accuracy:** The proportion of correctly classified test samples.
- **Precision:** The proportion of predicted samples for a class that actually belong to that class.
- **Recall:** The proportion of actual samples of a class correctly identified by the model.
- **F1-score:** The harmonic mean of precision and recall.

Macro-averaged precision, recall, and F1-score are used to assess performance across all three classes while giving each class equal weight.

Confusion matrices and visualizations are also used to examine classification performance and identify patterns in misclassification.

## 8. Results and Performance Comparison

The following table summarizes the recorded model evaluation results.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-score |
|---|---:|---:|---:|---:|
| Support Vector Machine (SVM) | 91.08% | 91.06% | 91.08% | 91.04% |
| Logistic Regression | 90.08% | 90.07% | 90.08% | 90.07% |
| K-Nearest Neighbors (kNN) | 88.58% | 88.65% | 88.58% | 88.47% |
| Multinomial Naive Bayes | 88.17% | 88.27% | 88.17% | 88.11% |
| Random Forest | 85.75% | 85.75% | 85.75% | 85.55% |
| Decision Tree | 70.08% | 70.21% | 70.08% | 70.13% |

*Note: These are the recorded results from the project's evaluation. Results may vary if the dataset, preprocessing settings, random split, or model configuration changes.*

### Interpretation

Based on the recorded results, **Support Vector Machine (SVM)** achieved the highest accuracy of 91.08% and a macro F1-score of 91.04% among the six evaluated models.

Logistic Regression was the second-best model by accuracy, achieving 90.08%. Both models performed well on this TF-IDF-based text-classification task.

The Decision Tree classifier achieved the lowest accuracy among the evaluated models. This indicates that, with the current features and configuration, it was less effective than the other classifiers for this dataset.

Overall, SVM was selected as the best-performing model based on the recorded evaluation results.

## 9. Visualizations

The main classification notebook includes evaluation visualizations to help interpret and compare the models.

These include:

- Comparison of classification accuracy across the six models.
- Confusion matrices showing actual versus predicted categories.
- Classification metrics used to compare model performance.

These visualizations make it easier to understand the relative strengths of the classifiers and identify which categories may be confused with one another.

## 10. Interactive Demo

The main classification notebook includes an interactive demo for testing the trained classification system.

The demo allows a user to provide research-paper data and obtain predicted academic categories. The predicted label indicates whether the abstract is classified as Science, Business, or Social Science.

A Sample-research-paper.csv file is also provided in the project folder for testing the demo.

The demo is intended to demonstrate how the trained model can be used to classify new research abstracts after the training and evaluation steps have been completed.

## 11. Project Structure

The project folder contains the following main files:

```text
Python_Classification_Project/
│
├── Research_Abstract_Classification.ipynb
├── Research_Abstract_Classification_Dataset_Creation.ipynb
├── research_abstracts_6000_cleaned.csv
├── README.md
└── Sample-research-paper.csv
```

**File descriptions:**

- `Research_Abstract_Classification.ipynb`: Main notebook containing preprocessing, feature extraction, model training, evaluation, visualizations, and the interactive demo.
- `Research_Abstract_Classification_Dataset_Creation.ipynb`: Notebook containing the dataset collection and preparation workflow.
- `research_abstracts_6000_cleaned.csv`: Finalized dataset used for training and evaluation.
- `README.md`: Project overview, methodology, results, and instructions.
- `Sample-research-paper.csv`: A separate CSV file containing sample research-paper data that can be used to test the interactive demo.

If model artifacts are saved separately, such as the trained SVM classifier and TF-IDF vectorizer, they can also be included in the project folder.

## 12. How to Run the Project

### Option 1: Google Colab

1. Open Google Colab.
2. Upload `Research_Abstract_Classification.ipynb`.
3. Upload `research_abstracts_6000_cleaned.csv` to the Colab session, or place it in the appropriate accessible location.
4. Check that the dataset path in the notebook matches the location of the uploaded CSV file.
5. Run the notebook cells in order.
6. Review the model evaluation results and visualizations.
7. Use the interactive demo to test research-paper abstracts or the supplied `Sample-research-paper.csv` file.

### Option 2: Jupyter Notebook

1. Install Python and Jupyter Notebook.
2. Install the required libraries if they are not already available:

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
   ```

3. Place the notebook and finalized dataset in the appropriate project directory.
4. Start Jupyter Notebook and open `Research_Abstract_Classification.ipynb`.
5. Run the cells in sequence.
6. Use the evaluation outputs and interactive demo.

### Running the Dataset Creation Notebook

The dataset creation notebook can be used to reproduce the data collection workflow. Depending on the current OpenAlex API requirements and the notebook's implementation, an API key is required to collect the full dataset. So kindly use your free openalex API key to recreate the dataset.

The supplied finalized CSV allows the main classification notebook to be run without recollecting the dataset.


## 13. Limitations

- The classification system is trained on the provided dataset and may not generalize equally well to every academic discipline or writing style.
- Research abstracts may contain interdisciplinary content that does not fit neatly into a single category.
- Predictions depend on the quality, coverage, and labeling of the training data.
- The evaluation results reflect the particular dataset split and model configurations used in the project.
- The model predicts one of the three predefined categories; it does not establish the scientific validity or quality of a research paper.

## 14. Conclusion

This project demonstrates the application of natural language processing and machine learning to multiclass research-abstract classification.

Six classifiers were trained and evaluated using TF-IDF features and standard classification metrics. Among the evaluated models, Support Vector Machine achieved the highest recorded accuracy of 91.08%, making it the best-performing model in this comparison.

The project also includes visualizations and an interactive demo to demonstrate the practical use of the classification system. The finalized dataset and sample research-paper CSV file are provided so that the notebook and demo can be tested without repeating the entire data collection process.

## 15. Sample Dataset for Testing the Demo

In addition to the main dataset, **the project folder also contains a separate CSV file named `Sample-research-paper.csv`**. This file can be used to test the interactive demo included in `Research_Abstract_Classification.ipynb`.

To test the demo, open the main notebook, navigate to the demo section, and provide the sample CSV file using the upload or input method supported by the demo. The system can then generate predicted academic categories for the supplied research-paper abstracts.

This sample file is intended for demonstration and testing purposes. The finalized dataset, `research_abstracts_6000_cleaned.csv`, remains the dataset used for the project's model training and evaluation.
