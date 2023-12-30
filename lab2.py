import pandas as pd
import re


def read_spambase(filename, skiprows, nrows):
    df = pd.read_csv(filename, skiprows = list(range(1, skiprows)), nrows = nrows, encoding = "latin-1")
    df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1, inplace = True)
    df = df.rename(columns = {"v1": "class", "v2": "text"})
    return df


def text_to_words(text):
    text = re.sub(r"\W+", " ", text)
    text = text.lower()
    words = text.split()
    return words


def count_letters(df):
    nspam = df["class"].value_counts().loc["spam"]
    nham = df["class"].value_counts().loc["ham"]
    print(f"Spam {nspam}")
    print(f"Ham {nham}\n")


def process_letters(df):
    letters = []

    for index, letter in df.iterrows():
        words = text_to_words(letter["text"])
        if not words:
            continue

        spam = False
        if letter["class"] == "spam":
            spam = True

        word_freqs = {}
        for word in words:
            if word not in word_freqs.keys():
                word_freqs[word] = 1
            else:
                word_freqs[word] += 1

        letters.append([spam, word_freqs])

    return letters


def calculate_k(df):
    k = len(df) ** 0.5
    k = int(k)
    if k % 2 == 0:
        k += 1
    return k


def euclid_difference(train_letter, test_letter):
    train_freqs = train_letter[1]
    test_freqs = test_letter[1]

    if len(train_freqs) < len(test_freqs):
        train_freqs, test_freqs = test_freqs, train_freqs

    total = 0
    for word in train_freqs.keys():
        test_freq = 0
        if word in test_freqs.keys():
            test_freq = test_freqs[word]
        total += (train_freqs[word] - test_freq) ** 2

    return total ** 0.5


def knn(k, train_letters, test_letters):
    spam_nletters = 0
    total_nletters = len(test_letters)

    guessed_spam_nletters = 0
    guessed_ham_nletters = 0

    for test_letter in test_letters:
        diffs = []
        for train_letter in train_letters:
            diff = euclid_difference(train_letter, test_letter)
            diffs.append((train_letter[0], diff))
        diffs = sorted(diffs, key = lambda letter: letter[1])
        diffs = diffs[:k]

        spam_count = 0
        ham_count = 0
        for diff in diffs:
            if diff[0]:
                spam_count += 1
            else:
                ham_count += 1
        guessed_spam = False
        if spam_count > ham_count:
            guessed_spam = True

        spam = test_letter[0]
        if spam:
            spam_nletters += 1

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

    nrows = 1000
    train_nrows = int(nrows * 0.8)
    train_df = read_spambase(filename, 0, train_nrows)
    count_letters(train_df)

    train_letters = process_letters(train_df)
    k = calculate_k(train_df)

    test_nrows = int(nrows * 0.2)
    test_df = read_spambase(filename, train_nrows, test_nrows)
    count_letters(test_df)

    test_letters = process_letters(test_df)
    knn(k, train_letters, test_letters)

    exit(0)


if __name__ == "__main__":
    main()
