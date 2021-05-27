import pandas as pd
import os
from nltk.corpus import opinion_lexicon
from src.main.python.common.common_functions import clean_text
import copy
from sklearn.feature_extraction.text import CountVectorizer

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


def estimate_sentiment_using_DTM(input_df, raw_text_col_name, min_words_thresh=100):
    """
    Estimate sentiment using a document term matrix

    Args:
        input_df: <pd.DataFrame> containing rows of raw text to process as a column
        raw_text_col_name: <str> column name of column containing the raw text strings
        min_words_thresh: <int> minimum number of cleaned words to include in final result, inclusive >=

    Returns:
        <dict> containing the processed sentiment dataframe, and positive and negative doc term matrices, representing
        the word frequencies per df row
    """
    CLEANED_TEXT = 'cleaned_text'
    CLEANED_TEXT_STR = 'cleaned_text_str'
    NUM_CLEANED_WORDS = 'num_cleaned_words'

    df = copy.deepcopy(input_df)

    df[CLEANED_TEXT] = df[raw_text_col_name].apply(lambda x: clean_text(x, text_str=True))
    df[NUM_CLEANED_WORDS] = df[CLEANED_TEXT].apply(lambda x: len(x))

    df = df[df[NUM_CLEANED_WORDS] >= min_words_thresh]

    sentiment_lexicons = {}
    sentiment_lexicons['positive_sentiment'] = positive_lexicon
    sentiment_lexicons['negative_sentiment'] = negative_lexicon

    sentiment_ls = list(sentiment_lexicons.keys())
    DTM_dict = {}

    df[CLEANED_TEXT_STR] = df[CLEANED_TEXT].apply(lambda x: ' '.join(x))

    for sentiment in sentiment_ls:
        count_vec = CountVectorizer(vocabulary=sentiment_lexicons[sentiment])
        dtm = count_vec.fit_transform(df[CLEANED_TEXT_STR])
        df_dtm = pd.DataFrame(dtm.toarray())
        df_dtm.columns = count_vec.vocabulary_.keys()
        DTM_dict[sentiment] = df_dtm
        df['{}_count'.format(sentiment)] = df_dtm.sum(axis=1)
        df['phi_{}'.format(sentiment)] = df['{}_count'.format(sentiment)] / df[NUM_CLEANED_WORDS]

    df['net_positive_tone'] = (df['phi_positive_sentiment'] - df['phi_negative_sentiment']) / (
            df['phi_positive_sentiment'] + df['phi_negative_sentiment'])

    return {'sentiment_dataframe': df,
            'DTM_pos': DTM_dict['positive_sentiment'],
            'DTM_neg': DTM_dict['negative_sentiment']}


if __name__ == '__main__':
    df = pd.read_csv(
        r"C:\Users\tkdmc\Documents\GitHub\personal_python\investment_analysis_with_nlp\S5 - Estimating Firm Level Sentiment\mda_data.csv",
        index_col=0)

    print(estimate_sentiment_using_DTM(df, raw_text_col_name='raw_text'))
