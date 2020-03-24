import pandas as pd
import numpy as np
import spacy
from spacy import displacy
from spacy.util import minibatch, compounding
import matplotlib.pyplot as plt
import plac
import random
from pathlib import Path

rain_data=pd.read_csv('rain.csv')
rain_data.shape

rain_data.head()

train_df = rain_data[['text','score']].dropna()

ax=train_df.score.value_counts().plot(kind='bar')
fig = ax.get_figure()

train_df['tuples'] = train_df.apply(
    lambda row: (row['text'],row['score']), axis=1)
train = train_df['tuples'].tolist()
train[:1]


def load_data(limit=0, split=0.8):
    train_data = train
    np.random.shuffle(train_data)
    train_data = train_data[-limit:]
    texts, labels = zip(*train_data)
    cats = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} for y in labels]
    split = int(len(train_data) * split)
    # split the data to training and testing
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label == "NEGATIVE":
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}


def train_model(model=None, output_dir=None, n_iter=10, init_tok2vec=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

    if model is not None:
        nlp = spacy.load("it_core_news_sm")  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe("textcat")

    # add label to text classifier
    textcat.add_label("POSITIVE")
    textcat.add_label("NEGATIVE")

    (train_texts, train_cats), (dev_texts, dev_cats) = load_data()
    # print(train_texts)
    # train_texts = train_texts[:n_texts]
    # print(train_texts.shape())

    # train_cats = train_cats[:n_texts]
    print(
        "Using {} examples ({} training, {} evaluation)".format(
            len(train_texts)+ len(dev_texts), len(train_texts), len(dev_texts)
        )
    )
    train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        if init_tok2vec is not None:
            with init_tok2vec.open("rb") as file_:
                textcat.model.tok2vec.from_bytes(file_.read())
        print("Training the model...")
        print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
        batch_sizes = compounding(4.0, 32.0, 1.001)
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            random.shuffle(train_data)
            batches = minibatch(train_data, size=batch_sizes)
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print(
                "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
                    losses["textcat"],
                    scores["textcat_p"],
                    scores["textcat_r"],
                    scores["textcat_f"],
                )
            )

    if output_dir is not None:
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
            print("Saved model to", output_dir)


def testing():
    (train_texts, train_cats), (dev_texts, dev_cats) = load_data()
    # print(dev_texts)
    # print(dev_cats)
    correct_prediction = []
    print("Loading model")
    nlp2 = spacy.load('italian_rain_model')
    for i in range(len(dev_texts)):
        tweet = dev_texts[i]
        label = dev_cats[i]
        doc = nlp2(tweet)
        if doc.cats['POSITIVE'] >= 0.5:
            if label['POSITIVE']:
                correct_prediction.append(True)
            else:
                continue
        else:
            if not label['POSITIVE']:
                correct_prediction.append(True)
            else:
                continue
    print("accuracy of prediction:", len(correct_prediction)/len(dev_texts)*1.0)
    train_true = []
    for temp in train_cats:
        if temp['POSITIVE']:
            train_true.append("yes")
        else:
            continue
    testing_true = []
    for temp in dev_cats:
        if temp['POSITIVE']:
            testing_true.append("yes")
        else:
            # print(temp)
            continue

    print("total number", len(train_texts) + len(dev_texts))
    print("training: ", len(train_texts), "positive: ", len(train_true), "nagtive: ",len(train_texts)-len(train_true))
    print("testing: ", len(dev_texts), "positive: ", len(testing_true), "nagtive: ", len(dev_texts)-len(testing_true))



def main():
    # print(load_data())
    # train_model(output_dir='italian_rain_model')
    testing()
    # print("Loading from model")
    # nlp2 = spacy.load('italian_rain_model')

if __name__== "__main__":
  main()
#
# test_text1 = 'RT @Culonainch: Inatteso Nubifragio, dramma maltempo, alluvione, violenti piogge, allagamenti, diluvio. Da noi in Cermania si kiama autunno'
# test_text2="RT @GuernseyJuliet: #ImpressioniDiSettembre Piove e allora...cosÃ¬ B'Domenica #DonneAmiche Guy Rose ðŸ‡ºðŸ‡¸ Marguerite, c.1909â€¦"
# doc = nlp2(test_text1)
# test_text1, doc.cats
#
# """## Positive review is indeed close to 1"""
#
# doc2 = nlp2(test_text2)
# test_text2, doc2.cats
#
# """## Negative review is close to 0"""

