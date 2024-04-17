import datetime
import pandas as pd
import streamlit as st
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "nlptown/bert-base-multilingual-uncased-sentiment"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "nlptown/bert-base-multilingual-uncased-sentiment"
    )
    return tokenizer, model


def sentiment_score(review, tokenizer, model):
    tokens = tokenizer.encode(review, return_tensors="pt")
    result = model(tokens)
    return int(torch.argmax(result.logits) + 1)


def main():
    st.title("OpenAI sentiment analysis using Bert NLP")
    csv_file = st.file_uploader(
        "Please upload Your Business' Reviews",
        type=["csv"],
        accept_multiple_files=False,  # Allow only one file to be uploaded
        help="Only CSV files are allowed.",
    )

    if csv_file is not None:
        if not csv_file.name.endswith(".csv"):
            st.error("Error: File not supported. Please upload a CSV file")
            return
        try:
            df = pd.read_csv(csv_file, encoding="utf-8")
        except UnicodeDecodeError:
            st.error(
                "Error: Unable to decode the CSV file. Please ensure it is UTF-8 compatible."
            )
            return
        except pd.errors.ParserError:
            st.error(
                "Error: Unable to parse the CSV file. Please check the file format."
            )
            return

        df[
            "Sentiment Score"
        ] = 0  # Add the 'Sentiment Score' column with default values
        snapshot = df.head(3)
        st.dataframe(snapshot)

        # Load the model
        tokenizer, model = load_model()

        # Press this button to see results
        if st.button("Analyze"):
            # Perform sentiment analysis on the 'Review' column
            df["Sentiment Score"] = df.iloc[:, 0].apply(
                lambda x: sentiment_score(x[:1200], tokenizer, model)
                if pd.notna(x)
                else 0
            )

            # Map sentiment scores to overall sentiment categories
            sentiment_mapping = {
                5: "Positive",
                4: "Positive",
                3: "Neutral",
                1: "Negative",
                2: "Negative",
            }
            df["Overall"] = df["Sentiment Score"].map(sentiment_mapping)

            # Display the updated DataFrame
            st.dataframe(df.head())

            csv_file_name = (
                datetime.datetime.now().strftime("%d-%m-%y") + "_updated_reviews.csv"
            )
            csv = df.to_csv(index=False)

            st.download_button(
                label="Download CSV File",
                data=csv,
                file_name=csv_file_name,
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
