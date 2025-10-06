
"""
Resume Matcher
- Compare resumes with a job description
- Rank candidates based on similarity
Dependencies: pip install scikit-learn pandas nltk
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re

# Download NLTK stopwords if not already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------------
# Helper Functions
# -------------------------

def clean_text(text):
    """Lowercase, remove special characters, numbers, and stopwords"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def read_resume(file_path):
    """Read resume file (txt)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return clean_text(text)

# -------------------------
# Main Matcher Function
# -------------------------

def match_resumes(resume_folder, job_description_file):
    """
    Inputs:
    - resume_folder: folder containing resumes (.txt)
    - job_description_file: path to job description (.txt)
    
    Returns:
    - DataFrame with resume filename and similarity score
    """
    # Read job description
    with open(job_description_file, 'r', encoding='utf-8') as f:
        job_desc = clean_text(f.read())

    # Read all resumes
    resumes = []
    filenames = []

    for file in os.listdir(resume_folder):
        if file.endswith('.txt'):
            filepath = os.path.join(resume_folder, file)
            text = read_resume(filepath)
            resumes.append(text)
            filenames.append(file)

    # Vectorize resumes and job description
    vectorizer = TfidfVectorizer()
    all_docs = resumes + [job_desc]
    tfidf_matrix = vectorizer.fit_transform(all_docs)

    # Compute cosine similarity between job description and resumes
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

    # Create DataFrame to display results
    df = pd.DataFrame({
        'Resume': filenames,
        'Match Percentage': [round(score*100, 2) for score in similarities]
    })

    df = df.sort_values(by='Match Percentage', ascending=False).reset_index(drop=True)
    return df

# -------------------------
# Example Usage
# -------------------------

if __name__ == "__main__":
    resume_folder = "resumes"  # folder with resumes in .txt
    job_description_file = "job_description.txt"

    if not os.path.exists(resume_folder):
        print(f"Resume folder '{resume_folder}' not found.")
    elif not os.path.exists(job_description_file):
        print(f"Job description file '{job_description_file}' not found.")
    else:
        result_df = match_resumes(resume_folder, job_description_file)
        print("Resume Matching Results:\n")
        print(result_df)
        # Optionally save to CSV
        result_df.to_csv("resume_matching_results.csv", index=False)
        print("\nResults saved to 'resume_matching_results.csv'.")
