import pandas as pd
import os
from nltk.corpus import opinion_lexicon
from src.main.python.common.common_functions import clean_text

# Use NLTK lexicons
positive_lexicon = set(opinion_lexicon.positive())
negative_lexicon = set(opinion_lexicon.negative())


def estimate_sentiment(path_to_folder):
    """
    Estimate the sentiment per file in folder of text files

    Args:
        path_to_folder: <str> path to the folder containing text files to evaluate

    Returns:
        <pd.DataFrame> summarizing sentiment scores per file
    """
    files_ls = os.listdir(path_to_folder)
    files_ls = [file for file in files_ls if file.endswith('.txt')]  # only get the text files

    sentiment_scores = {}

    for file in files_ls:
        # Estimate Sentiment
        cleaned_words = clean_text('{}/{}'.format(path_to_folder, file))

        # only compute for files with more than 250 cleaned words
        if len(cleaned_words) < 250:
            continue

        positive_sentiment = len([word for word in cleaned_words if word in positive_lexicon])
        negative_sentiment = len([word for word in cleaned_words if word in negative_lexicon])

        phi_pos = positive_sentiment / len(cleaned_words)
        phi_neg = negative_sentiment / len(cleaned_words)

        phi_npt = (phi_pos - phi_neg) / (phi_pos + phi_neg)

        sentiment_scores[file] = [phi_pos, phi_neg, phi_npt, len(cleaned_words)]

    df_scores = pd.DataFrame(sentiment_scores).T
    df_scores.reset_index(inplace=True)
    df_scores.columns = ['file_name', 'phi_pos', 'phi_neg', 'phi_npt', 'num_cleaned_words']

    return df_scores


if __name__ == '__main__':
    print(estimate_sentiment(
        r'C:\Users\tkdmc\Documents\GitHub\personal_python\investment_analysis_with_nlp\S5 - Estimating Firm Level Sentiment\mda').head())
