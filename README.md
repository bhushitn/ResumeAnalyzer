# Intelligent Resume Analyzer & Classifier using Gemini API

This project demonstrates how to leverage Google's Gemini Generative AI models to automate the processing and classification of resumes, addressing common challenges faced by HR departments and recruitment agencies.

## Table of Contents

1.  [Problem Statement](#problem-statement)
2.  [Proposed Solution](#proposed-solution)
3.  [Methodology](#methodology)
4.  [Data](#data)
5.  [Assumptions](#assumptions)
6.  [Setup & Usage](#setup--usage)
7.  [Exploratory Data Analysis (EDA) & Visualization](#exploratory-data-analysis-eda--visualization)
8.  [Modeling](#modeling)
9.  [Results](#results)
10. [Interpretation](#interpretation)
11. [Recommendations & Future Work](#recommendations--future-work)
12. [Acknowledgements](#acknowledgements)

## Problem Statement

Human Resources departments and recruitment firms often receive a large volume of resumes daily. Manually:

*   Reading through each resume to identify key qualifications, experience, and skills.
*   Extracting this information into a structured format for comparison or database entry.
*   Categorizing resumes into relevant job roles or talent pools.

...is a time-consuming, repetitive, and potentially inconsistent process. This inefficiency can lead to delays in hiring, overlooked candidates, and increased operational costs.

## Proposed Solution

This project implements a two-part solution using Google's Gemini API to automate and enhance resume processing:

1.  **AI-Powered Resume Structuring:** Utilizes Gemini's ability to understand context and follow instructions (few-shot prompting) combined with its JSON output mode to parse raw resume text (`Resume_str`) into a structured JSON object. This extracts key fields like `job_title`, `years_experience`, `summary`, `skills`, `education`, `experience`, etc., making the information readily accessible for databases and analysis.
2.  **Automated Resume Classification:** Employs Gemini's text embedding capabilities (`text-embedding-004`) to convert the semantic meaning of resume text into numerical vectors. These embeddings are then used to train a neural network (using Keras/TensorFlow) to automatically classify resumes into predefined job categories (e.g., 'HR', 'ENGINEERING', 'SALES', 'FITNESS').

## Methodology

*   **Data Loading & Preparation:** Load the dataset, handle missing values, filter based on selected categories, and limit the number of samples per category if needed.
*   **Resume Structuring (Optional):**
    *   Define a target JSON schema using Python's `TypedDict`.
    *   Craft a few-shot prompt with clear instructions and diverse input/output examples.
    *   Iterate through resumes, calling the Gemini API (`gemini-1.5-flash` or `gemini-1.5-pro`) with the prompt and schema to generate structured JSON.
    *   Implement robust error handling for API calls and JSON parsing.
*   **Resume Classification:**
    *   Prepare data: Select raw resume text and corresponding category labels. Encode labels numerically. Perform a stratified train-test split.
    *   Generate Embeddings: Use the Gemini API (`text-embedding-004` with `RETRIEVAL_DOCUMENT` task type) to create vector representations of the resume text for both training and test sets. Handle potential API errors during embedding generation.
    *   Build Classifier: Construct a feed-forward neural network using Keras with embedding input, hidden layers (Dense, Dropout), and a softmax output layer.
    *   Train Classifier: Train the Keras model on the generated embeddings, using early stopping based on validation loss to prevent overfitting.
    *   Evaluate Classifier: Assess the trained model's performance on the unseen test set using accuracy, precision, recall, F1-score, and a confusion matrix.
*   **Prediction Pipeline:** Create a function that takes new, raw resume text, generates its embedding, and uses the trained classifier to predict the job category.
*   **Retry Logic:** Implement automatic retries for Gemini API calls to handle transient issues like rate limits or server errors.

## Data

*   **Source:** [Resume Dataset on Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) by Snehaan Bhawal.
*   **Content:** Contains over 2400 resumes in CSV format with columns:
    *   `ID`: Unique identifier.
    *   `Resume_str`: Raw resume text.
    *   `Resume_html`: HTML version (not used in this project).
    *   `Category`: Predefined job category (24 categories available).
*   **Subsetting:** The notebook allows for filtering by specific `SELECTED_JOB_CATEGORIES` and limiting the `MAX_RESUMES_PER_CATEGORY` for focused analysis or faster processing/training.

## Assumptions

*   **API Access:** Valid Google AI API key with sufficient quota for Gemini API calls is required.
*   **Data Quality:** The input resume text (`Resume_str`) is assumed to be reasonably clean and representative of typical resumes. The quality of the predefined `Category` labels is also assumed to be adequate for training.
*   **Category Relevance:** The selected job categories are assumed to be distinct enough for meaningful classification based on resume content.
*   **Schema Suitability:** The defined `StructuredResumeSchema` is assumed to capture the most relevant information for typical HR/recruitment use cases.
*   **Environment:** The code is designed to run in an environment with Python 3.x and the libraries listed in `requirements.txt` (implicitly installed in the notebook setup).

## Setup & Usage

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Install Dependencies:** The notebook installs required libraries using `pip`. If running locally outside the notebook, you might need:
    ```bash
    pip install google-generativeai python-dotenv pandas scikit-learn tensorflow matplotlib seaborn tqdm rich jupytext nbformat ipykernel
    ```
3.  **Configure API Key:**
    *   **Recommended:** Create a `.env` file in the project root directory with the line: `GOOGLE_API_KEY=YOUR_API_KEY_HERE`
    *   Alternatively, configure it using Kaggle Secrets or Colab Secrets as described in the notebook's comments (Section 1.3).
4.  **Configure Parameters:** Adjust settings in the notebook (Section 1.2), especially:
    *   `PROCESS_RAW_RESUMES`: Set to `True` for the first run to structure data using the API, or `False` to load previously saved structured data (`structured_resumes_output.csv`).
    *   `RAW_DATA_CSV_PATH`: Ensure this points to the correct location of the input `Resume.csv`.
    *   `SELECTED_JOB_CATEGORIES` / `MAX_RESUMES_PER_CATEGORY`: Modify if desired.
5.  **Run the Notebook:** Execute the cells sequentially in a Jupyter environment (like Jupyter Lab, VS Code, Google Colab, or Kaggle).

## Exploratory Data Analysis (EDA) & Visualization

*   **Raw Data:** The initial dataset contains 24 distinct job categories, with varying numbers of resumes per category. Some categories like 'Information-Technology' and 'Business-Development' are well-represented, while others like 'BPO' are less common. Basic checks for missing values in `Resume_str` are performed.
*   **Structured Data (if generated):**
    *   *Years of Experience:* A histogram shows the distribution of estimated experience, often skewed towards lower years or having peaks corresponding to common career milestones. Null values indicate difficulty in parsing experience duration.
    *   *Skills Analysis:* A bar chart visualizes the most frequent skills extracted across the processed resumes, highlighting common technologies, methodologies, and soft skills relevant to the analyzed categories.
*   **Classification Data:** Distribution plots confirm that the stratified train-test split maintains similar class proportions between the training and testing sets, which is crucial for unbiased model evaluation.

*(Refer to the plots generated within the notebook for specific visualizations.)*

## Modeling

### 1. Resume Structuring Model

*   **Approach:** Few-Shot Learning with Gemini API's JSON Mode.
*   **Model:** `gemini-1.5-flash` (configurable).
*   **Input:** Raw resume text (`Resume_str`).
*   **Prompt:** Includes system instructions, schema examples, 2-3 diverse input/output resume examples, and the target resume text.
*   **Output:** A JSON object adhering to the `StructuredResumeSchema`.
*   **Error Handling:** Logs API call errors and JSON parsing errors.

### 2. Resume Classification Model

*   **Approach:** Supervised learning using text embeddings as features.
*   **Feature Extraction:**
    *   Model: `text-embedding-004` (configurable).
    *   Task Type: `RETRIEVAL_DOCUMENT` (configurable).
    *   Input: Raw resume text (`Resume_str`).
    *   Output: 768-dimensional embedding vector per resume.
*   **Classifier:**
    *   Type: Feed-Forward Neural Network (Sequential Keras model).
    *   Architecture:
        *   Input Layer (shape=embedding dimension)
        *   Dense Hidden Layer 1 (ReLU activation, Dropout)
        *   Dense Hidden Layer 2 (ReLU activation, Dropout)
        *   Output Layer (Softmax activation, size=number of classes)
    *   Optimizer: Adam.
    *   Loss Function: Sparse Categorical Crossentropy.
*   **Training:** Trained on embedded training data, validated on embedded test data, using early stopping on validation loss.

## Results

---

## Interpretation



## Recommendations & Future Work


## Acknowledgements

*   **Google:** For the Gemini API and Generative AI resources.
*   **Kaggle:** For the platform and dataset hosting.

