import pandas as pd
import re


def read_spambase(filename, skiprows, nrows):
    df = pd.read_csv(filename, skiprows = list(range(1, skiprows)), nrows = nrows, encoding = "latin-1")
    df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1, inplace = True)
    df = df.rename(columns = {"v1": "class", "v2": "text"})
    return df


def count_letters(df):
    nspam = df["class"].value_counts().loc["spam"]
    nham = df["class"].value_counts().loc["ham"]
    print(f"Spam {nspam}")
    print(f"Ham {nham}\n")


def text_to_words(text):
    text = re.sub(r"\W+", " ", text)
    text = text.lower()
    words = text.split()
    return words


def get_freqs(df):
    spam_freqs = {}
    ham_freqs = {}
    spam_nletters = 0
    total_nletters = len(df)
    for index, letter in df.iterrows():
        words = text_to_words(letter["text"])
        if not words:
            continue

        spam = False
        if letter["class"] == "spam":
            spam = True
            spam_nletters += 1

        for word in words:
            if spam:
                if word not in spam_freqs.keys():
                    spam_freqs[word] = 1
                else:
                    spam_freqs[word] += 1
            else:
                if word not in ham_freqs.keys():
                    ham_freqs[word] = 1
                else:
                    ham_freqs[word] += 1

    ham_nletters = total_nletters - spam_nletters
    ps = spam_nletters / total_nletters  # P(S)
    ph = ham_nletters / total_nletters  # P(H)

    return [spam_freqs, ham_freqs, ps, ph]


def get_probs(freqs):
    nwords = sum(freqs.values())

    probs = {}
    for word, freq in freqs.items():
        probs[word] = freq / nwords
    return probs


def get_weights(df, spam_probs, ham_probs, ps, ph):
    spam_nletters = 0
    total_nletters = len(df)

    guessed_spam_nletters = 0
    guessed_ham_nletters = 0

    for index, letter in df.iterrows():
        words = text_to_words(letter["text"])
        if not words:
            continue

        spam = False
        if letter["class"] == "spam":
            spam = True
            spam_nletters += 1

        weights = []
        for word in words:
            pws = 0
            if word in spam_probs.keys():
                pws = spam_probs[word]
            pwh = 0
            if word in ham_probs.keys():
                pwh = ham_probs[word]

            psw = 0
            if pwh != 0 or pws != 0:
                psw = (pws * ps) / (pws * ps + pwh * ph)
            weights.append(psw)

        p = sum(weights) / len(weights)
        p = round(p * 100, 2)

        limit = 30
        guessed_spam = False
        if p > limit:
            guessed_spam = True

        if spam == guessed_spam == True:
            guessed_spam_nletters += 1
        if spam == guessed_spam == False:
            guessed_ham_nletters += 1

    spam_detection_accuracy = guessed_spam_nletters / spam_nletters
    spam_detection_accuracy = round(spam_detection_accuracy * 100, 2)
    print(f"Spam detected in {spam_detection_accuracy}% of cases")

    ham_nletters = total_nletters - spam_nletters
    ham_detection_accuracy = guessed_ham_nletters / ham_nletters
    ham_detection_accuracy = round(ham_detection_accuracy * 100, 2)
    print(f"Ham detected in {ham_detection_accuracy}% of cases")


def main():
    pd.set_option("display.max_colwidth", None)
    filename = "spambase/spam.csv"

    nrows = 5000
    train_nrows = int(nrows * 0.8)
    train_df = read_spambase(filename, 0, train_nrows)
    count_letters(train_df)

    spam_freqs, ham_freqs, ps, ph = get_freqs(train_df)
    spam_probs = get_probs(spam_freqs)
    ham_probs = get_probs(ham_freqs)


    test_nrows = int(nrows * 0.2)
    test_df = read_spambase(filename, train_nrows, test_nrows)
    count_letters(test_df)

    get_weights(test_df, spam_probs, ham_probs, ps, ph)

    exit(0)


if __name__ == "__main__":
    main()
