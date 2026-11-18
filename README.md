# Arabic Dialect Identification

[![tests](https://github.com/eslammohibrahim/arabic-dialect-id/actions/workflows/tests.yml/badge.svg)](https://github.com/eslammohibrahim/arabic-dialect-id/actions/workflows/tests.yml)

As an Egyptian I can usually tell where someone is from after one sentence. I wanted to know if a plain linear model could do the same, so I trained one on Arabic tweets from five countries: Egypt, Libya, Lebanon, Sudan and Morocco.

It got 86.5% of the test tweets right (14,753 tweets, macro F1 0.836). That number needs some context though. Egyptian tweets are 39% of the data, so a "model" that answers Egyptian every time already gets 39% without reading anything. That's why I compared models with macro F1 instead.

```text
$ python src/predict.py
Egyptian  | انا مش فاهم انت عايز ايه بالظبط
Lebanese  | شو عم تعمل هلق؟ بدك نطلع سوا
Libyan    | شن تبي توا؟ نبي نمشي للسوق
Moroccan  | واش نتا بخير؟ بغيت نشوفك غدا
Sudanese  | انت داير شنو؟ زول ما عندو موضوع
```

I made up these sentences myself and yes, they're the easy kind. Real tweets are messier.

## Data

[Arabic Dialectical Tweets](https://huggingface.co/datasets/amgadhasan/arabic_tweets_dialects) from Hugging Face, MIT license, 147,725 tweets with a country label each. It's not in the repo; `src/data.py` grabs it on the first run.

| Dialect | Share of train |
|---|---|
| Egyptian | 39.0% |
| Libyan | 24.7% |
| Lebanese | 18.7% |
| Sudanese | 9.7% |
| Moroccan | 7.8% |

I reused the text normalization from my [sentiment project](https://github.com/eslammohibrahim/arabic-sentiment-analysis) (tashkeel, the different alefs, stretched letters, mentions, links). After that I threw out 116 tweets with less than 3 words, 44 that showed up under two different countries, and 44 duplicates. Honestly expected way more duplicates from a Twitter dataset. 147,521 left, split 80/10/10.

## Models

TF-IDF on words (1-2 grams) or on characters (2-5 grams), and each model got a few values of its main setting tried on validation.

| Model | Val accuracy | Val macro F1 |
|---|---|---|
| Always Egyptian | 39.0% | 0.112 |
| Naive Bayes, words | 84.8% | 0.815 |
| Logistic Regression, words | 84.5% | 0.814 |
| Linear SVM, words | 84.6% | 0.815 |
| Linear SVM, characters | 85.5% | 0.825 |
| Linear SVM, words + characters | 86.3% | 0.835 |

Naive Bayes being as good as logistic regression surprised me a bit, and it trains in under a second.

Characters did better than words. My guess: dialects differ a lot in little prefixes stuck to the verb. Egyptians say بيعمل, Lebanese say عم يعمل, and on Twitter nobody spells anything the same way twice, so whole words are noisy. The combined one won, I retrained it on train+val and ran the test set one time.

## Test results

| Dialect | Recall |
|---|---|
| Egyptian | 93.4% |
| Lebanese | 88.3% |
| Libyan | 85.6% |
| Moroccan | 72.2% |
| Sudanese | 68.8% |

![Confusion matrix](reports/figures/confusion_matrix.png)

Sudanese is the worst. 17% of Sudanese tweets get labeled Egyptian, and when I looked at them it's not crazy, the two share مش and كدا and دا, plus there are 4x more Egyptian tweets to learn from. Moroccan mistakes go to Egyptian and Libyan about equally. I haven't done anything about the imbalance yet, class weights are next.

## What it learned

Top words per dialect (word features only):

| Dialect | Top words |
|---|---|
| Egyptian | ده, يا عم, دول, عندك حق, علشان, ازاي, كام |
| Lebanese | عم, شو, هيك, هيدي, مبارح, رح, بدو |
| Libyan | شن, توا, هكي, هضا, راهو, لين |
| Moroccan | واش, حيت, هاد, ديال, ملي, اخويا |
| Sudanese | ديل, ياخ, دا, ساي, لامن, عديل |

Mostly the words you'd expect. The funny ones: مصر is one of the top Egyptian words, and the Moroccan list has الاتحاد النصر, which are football clubs. So the model partly learned what each country tweets about, not just how people talk. Write about Egyptian football in Lebanese and it'll probably get confused.

## Run it

```bash
pip install -r requirements.txt
python src/train.py      # downloads data, trains everything, saves models/dialect_model.joblib (few minutes on CPU)
python src/predict.py "ازيك عامل ايه"
pytest -q
```

## Limitations

- Five dialects only. A Saudi or Iraqi tweet still gets one of these five.
- No idea how the labels were made, the dataset card doesn't say. Some tweets are just MSA, and for those the label can only be guessed from the topic.
- Trained on tweets. Very short or very formal text is harder.

## License

MIT
