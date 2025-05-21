# Set seed for reproducibility
from neuralprophet import set_random_seed 
import os

seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)

# Libraries, modules and variables
from binance.client import Client
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import copy
import contractions 
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import html
from neuralprophet import NeuralProphet
import math
from matplotlib import pyplot as plt 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd 
import re
import requests
import seaborn as sns
import snscrape.modules.twitter as sntwitter 
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud

numLags = 3
forecast = 3
length = 150
client = Client()
tweets = []
queries = []
pricesDf = None
tweetsDf =  None
tag = "BTCUSDT"
searchTwitter = "bitcoin"
priceSheet = "BTCPrices.xlsx"
tweetSheet = "BTCTweets.xlsx"
mergedSheet = "BTCSentiment.xlsx"

abbreviations = {
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "ath" : "all time high",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "btw" : "by the way",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dyor" : "do your own research",
    "eg" : "example",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fb" : "facebook",
    "fig" : "figure",
    "fomo" : "fear of missing out",
    "fud" : "fear, uncertainty, and doubt",
    "fyi" : "for your information",
    "gbp" : "great britsh pound",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gm" : "good morning",
    "gn" : "good night",
    "gratz" : "congratulations",
    "hodl" : "hold on for dear life",
    "idc" : "i do not care",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ig" : "instagram",
    "iirc" : "if i remember correctly",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "irl" : "in real life",
    "jk" : "just kidding",
    "jpy" : "japanese yen",
    "jsyk" : "just so you know",
    "lfg" : "let us fucking go",
    "lmao" : "laughing my ass off",
    "lmfao" : "laughing my fucking ass off",
    "lol" : "laughing out loud",
    "nfa" : "not financial advice",
    "ngl" : "not going to lie",
    "ngmi" : "not going to make it",
    "nsfw" : "not safe for work",
    "nth" : "nothing",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "omg" : "oh my god",
    "omw" : "on my way",
    "pls" : "please",
    "plz" : "please",
    "pov" : "point of view",
    "ppl" : "people",
    "rekt" : "wrecked",
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "srsly" : "seriously", 
    "tbh" : "to be honest",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "ttyl" : "talk to you later",
    "u" : "you",
    "usd" : "united states dollar",
    "us" : "united states",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wgmi" : "we are going to make it",
    "wnt" : "want",
    "wtf" : "what the fuck",
    "wth" : "what the hell",
    "wuzup" : "what is up",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
}

# Functions
def determine_sentiment(score):
    if(score >= 0.05):
        return "positive"
    elif(score <= -0.05):
        return "negative"
    else:
        return "neutral"

@st.cache_data
def develop_np_model(ar_layers, learning_rate, trainDfNp, testDfNp):
    modelNp = NeuralProphet(
        n_forecasts=forecast,
        n_lags=numLags,
        ar_layers=ar_layers,
        loss_func="MSE",
        epochs=150,
        learning_rate=learning_rate,
    )

    set_random_seed(seed)
    modelNp.add_lagged_regressor("regressor1")
    modelNp.add_lagged_regressor("regressor2")

    metrics = modelNp.fit(trainDfNp, freq="D", validation_df=testDfNp)
    return modelNp, metrics

def expand_con(tweet):
    word = contractions.fix(tweet)
    return word

def get_tweets():
    currentDate = date.today() + relativedelta(days=-1)
    targetDate = currentDate + relativedelta(years=-3) + relativedelta(days=1)
    # Return the tweets excel file. If not found, then create it
    try:
        tweetsDf = pd.read_excel(tweetSheet)
        tweetsDf["content"] = tweetsDf["content"].astype(str)
        tweetsDf["date"] = tweetsDf["date"].dt.date
    except FileNotFoundError:
        tweets = perform_queries(currentDate, targetDate)
        tweetsDf = pd.DataFrame(tweets)
        tweetsDf["date"] = tweetsDf["date"].dt.date
        tweetsDf.to_excel(tweetSheet, index=False)
    except Exception as e:
        print(f"Error occurred while reading {tweetSheet}: {e}")
        return None
    return tweetsDf

def get_prices():
    currentDate = date.today() + relativedelta(days=-1)
    targetDate = currentDate + relativedelta(years=-3) + relativedelta(days=1)
    # Return the price excel file. If not found, then create it
    try:
        pricesDf = pd.read_excel(priceSheet)
        pricesDf["openDate"] = pricesDf["openDate"].dt.date
        pricesDf["closeDate"] = pricesDf["closeDate"].dt.date
    except FileNotFoundError:
        pricesDf = retrieve_prices(currentDate, targetDate)
        pricesDf = preprocess_prices(pricesDf)
        pricesDf.to_excel(priceSheet, index=False)
        pricesDf[["open", "close", "volume"]] = pricesDf[["open", "close", "volume"]].astype(float)
    except Exception as e:
        print(f"Error occurred while reading {priceSheet}: {e}")
        return None
    return pricesDf

def generate_wordcloud(tweetsDf):
    # Getting the past 3 days' tweets
    stopWords = stopwords.words('english')
    tweetsForCloud = copy.deepcopy(tweetsDf)
    date = tweetsForCloud["date"].max()
    targetDate = date + relativedelta(days=-2)
    tweetsForCloud = tweetsForCloud[tweetsForCloud["date"].between(targetDate, date)]
    tweetsForCloud = tweetsForCloud.reset_index(drop="True")

    # Remove duplicates
    tweetsForCloud = tweetsForCloud.loc[~tweetsForCloud["content"].duplicated()].reset_index(drop=True)
    # Convert to lowercase
    tweetsForCloud["cleanedContent"] = tweetsForCloud["content"].str.lower()
    # Expand contractions
    tweetsForCloud["cleanedContent"] = tweetsForCloud["cleanedContent"].apply(expand_con)
    # Remove HTML entities
    tweetsForCloud["cleanedContent"] = tweetsForCloud["cleanedContent"].apply(replace_html_entities)
    # Remove escape characters
    tweetsForCloud["cleanedContent"] = tweetsForCloud["cleanedContent"].apply(remove_escape_chars)
    # Remove special characters
    tweetsForCloud["cleanedContent"] = tweetsForCloud["cleanedContent"].apply(remove_special_chars)
    # Replace abbreviations
    tweetsForCloud["cleanedContent"] = tweetsForCloud["cleanedContent"].apply(replace_abb)
    # Remove white spaces 
    tweetsForCloud["cleanedContent"] = tweetsForCloud["cleanedContent"].apply(lambda x: x.split())
    tweetsForCloud["cleanedContent"] = tweetsForCloud["cleanedContent"].apply(lambda x: ' '.join(x))
    # Tokenization
    tweetsForCloud["token"] = tweetsForCloud["cleanedContent"].apply(lambda x: nltk.word_tokenize(x))
    # Remove stopwords
    tweetsForCloud["tokenWithoutStopword"] = tweetsForCloud["token"].apply(lambda x: remove_stopwords(x, stopWords))
    tweetsForCloud["tokenWithoutStopword"] = tweetsForCloud["tokenWithoutStopword"].apply(lambda x: [i for i in x if len(i) > 2])
    # Lemmatization
    tweetsForCloud["lemmatisedToken"] = tweetsForCloud["tokenWithoutStopword"].apply(lemmatise_tokens)
    # Join the tokens in to a string
    tweetsForCloud["string"] = tweetsForCloud["lemmatisedToken"].apply(lambda x: ' '.join(x))
    # Join all the strings in to a single string
    totalString = ' '.join(tweetsForCloud["string"])
    
    wordCloud = WordCloud(width=3000, height=2000, background_color="white", min_font_size = 20, collocations=False).generate(totalString)
    plt.axis("off")
    plt.imshow(wordCloud)
    plt.savefig("wordcloud.png", bbox_inches='tight', pad_inches=0.1)
    st.image("wordcloud.png", use_column_width=True)

def generate_barplot_sentiment(tweetsDf):
    fig = plt.figure(figsize=(8, 5))
    lastDate = tweetsDf["date"].iloc[-1]
    targetDate = lastDate + relativedelta(days=-2)
    tweetsDf = tweetsDf[tweetsDf["date"] >= targetDate]
    tweetsDf = tweetsDf.loc[~tweetsDf["content"].duplicated()].reset_index(drop=True)
    tweetsDf["cleanedContent"] = tweetsDf["content"].apply(replace_html_entities)
    tweetsDf["cleanedContent"] = tweetsDf["cleanedContent"].apply(remove_escape_chars)
    tweetsDf["sentimentScore"] = tweetsDf["cleanedContent"].apply(perform_sentiment_analysis)
    tweetsDf["sentiment"] = tweetsDf["sentimentScore"].apply(determine_sentiment)
    countSentiment = Counter(tweetsDf["sentiment"])
    sns.barplot(x=list(countSentiment.keys()), y=list(countSentiment.values()))
    plt.xticks(fontsize=15)
    plt.xlabel("Sentiment", fontdict={"fontsize":20}, labelpad=20)
    plt.ylabel("Count", fontdict={"fontsize":20}, labelpad=20)
    st.pyplot(fig)

def generate_loss_plot(metrics):
    fig = plt.figure(figsize=(8, 5))
    sns.lineplot(data=metrics[["Loss", "Loss_val"]])
    plt.xlabel("Epochs", fontdict={"fontsize":15})
    plt.ylabel("Loss", fontdict={"fontsize":15})
    return fig

def generate_actual_vs_predicted_plot(errorStr, arrangedDf):
    fig = plt.figure(figsize=(8,5))
    sns.set(rc={'figure.figsize':(15,8)})
    sns.lineplot(data=arrangedDf[["actualPrice", "predictedPrice"]])
    plt.ylabel("Close Price", fontdict={"fontsize":15})
    plt.xlabel("Date", fontdict={"fontsize":15})
    plt.annotate(errorStr, xy=(0.775, 0.095), xycoords='axes fraction',
                bbox=dict(boxstyle="round", fc="white", ec="black", lw=2),
                arrowprops=dict(facecolor='black', arrowstyle='->'))
    return fig

def get_merged(pricesDf, tweetsDf):
    try:
        mergedDf = pd.read_excel(mergedSheet)
    except FileNotFoundError:
        averageScores = preprocess_and_perform_sentiment_analysis(tweetsDf)
        mergedDf = pd.merge(averageScores, pricesDf["close"], left_index=True, right_index=True)
        rearrangeColumns = ["date","averageSentimentScore","close"]
        mergedDf = mergedDf.reindex(columns=rearrangeColumns)
        mergedDf.to_excel(mergedSheet, index = False)
        mergedDf["close"] = mergedDf["close"].astype(float)
    except Exception as e:
        print(f"Error occurred while reading {mergedSheet}: {e}")
        return None
    return mergedDf

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        raise ValueError(f"Invalid URL {url}")
    return r.json()

def lemmatise_tokens(tokens):
    lemmatiser = WordNetLemmatizer()
    tags = nltk.pos_tag(tokens)
    lemmatised_tokens = []
    
    for token,tag in tags:
        if tag.startswith('J'): #Adjective
            pos = 'a'
        elif tag.startswith('V'): #Verb
            pos = 'v'
        elif tag.startswith('N'): #Noun
            pos = 'n'
        elif tag.startswith('R'): #Adverb
            pos ='r'
        else:
            pos = 'n'
        lemmatised = lemmatiser.lemmatize(token, pos=pos)
        lemmatised_tokens.append(lemmatised)
    return lemmatised_tokens

@st.cache_data
def reload_data():
    pricesDf = get_prices()
    tweetsDf = get_tweets()
    mergedDf = get_merged(pricesDf, tweetsDf)
    return pricesDf, tweetsDf, mergedDf

def remove_escape_chars(tweet):
    tweet = re.sub(r'[\x00-\x1F\x7F]',' ',tweet)
    return tweet

def remove_special_chars(tweet):
    tweet = re.sub(r'[^a-zA-Z\s]',' ',tweet)
    return tweet

def remove_stopwords(tokens, stopWords):
    check = [token for token in tokens if token.lower() not in stopWords]
    return check

def replace_abb(tweet):
    for abb, actual in abbreviations.items():
        pattern = r"\b" + abb + r"\b"
        tweet = re.sub(pattern, actual, tweet)
    return tweet

def replace_html_entities(tweet):
    tweet = html.unescape(tweet)
    return tweet

def retrieve_prices(currentDate, targetDate):
    btcPrices = client.get_historical_klines(tag, Client.KLINE_INTERVAL_1DAY, 
                                         str(targetDate), str(currentDate))
    
    pricesDf = pd.DataFrame(btcPrices, columns=["openTime", "open", "high", "low", "close", "volume", "closeTime", "quoteAssetVolume", "numberOfTrades", "takerBuyBaseVol", "takerBuyQuoteVol", "ignore"])
    return pricesDf

def preprocess_prices(pricesDf):
    pricesDf.rename(columns={"openTime":"openDate", "closeTime":"closeDate"}, inplace = True)
    pricesDf = pricesDf.drop("ignore", axis=1)
    pricesDf["openDate"] = pricesDf["openDate"].apply(lambda x: datetime.fromtimestamp(x / 1000.0).date())
    pricesDf["closeDate"] = pricesDf["closeDate"].apply(lambda x: datetime.fromtimestamp(x / 1000.0).date())
    return pricesDf

def preprocess_data_np(mergedDfNp):
    # Perform log transformation on close
    mergedDfNp["close"] = mergedDfNp["close"].apply(lambda x: math.log(x))

    # Shift the close 
    mergedDfNp["closeOfPreviousDay"] = mergedDfNp["close"].shift(1)
    rearrangeColumns = ["date","averageSentimentScore","closeOfPreviousDay", "close"]
    mergedDfNp = mergedDfNp.reindex(columns=rearrangeColumns)
    mergedDfNp.drop(mergedDfNp.index[0], inplace=True)

    # Rename the columns
    mergedDfNp = mergedDfNp.rename(columns={"date": "ds", "averageSentimentScore":"regressor1", 
                                            "closeOfPreviousDay": "regressor2", "close": "y"})
    
    # Split the data frame into training and testing data frames
    trainPercentage = 0.80

    numSamplesNp = len(mergedDfNp)

    numTrainNp = int(trainPercentage * numSamplesNp)

    trainDfNp = mergedDfNp.iloc[:numTrainNp]
    testDfNp = mergedDfNp.iloc[numTrainNp:]

    # Perform normalization
    scaler = MinMaxScaler()
    trainDfNp.loc[:,["regressor1", "regressor2"]] = scaler.fit_transform(trainDfNp[["regressor1", "regressor2"]])
    testDfNp.loc[:,["regressor1", "regressor2"]] = scaler.transform(testDfNp[["regressor1", "regressor2"]])

    return trainDfNp, testDfNp
    
def perform_queries(currentDate, targetDate):
    tweets = []
    for i in range(((currentDate-targetDate).days)+1):
        query = searchTwitter + " lang:en-GB -filter:mentions -filter:hashtags -filter:retweets -filter:links -filter:replies since:" + str(targetDate) + " until:" + str(targetDate + relativedelta(days=1))
        queries.append(query)
        targetDate = targetDate + relativedelta(days=1)

    with ThreadPoolExecutor(max_workers=30) as executor:  
        results = list(executor.map(scrape_tweets, queries, [length]*len(queries)))
        for result in results:
            tweets.extend(result)  
    return tweets

def perform_sentiment_analysis(tweet):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(tweet)["compound"]

def preprocess_and_perform_sentiment_analysis(tweetsDf):
    tweetsDf = tweetsDf.loc[~tweetsDf["content"].duplicated()].reset_index(drop=True)
    tweetsDf["cleanedContent"] = tweetsDf["content"].apply(replace_html_entities)
    tweetsDf["cleanedContent"] = tweetsDf["cleanedContent"].apply(remove_escape_chars)
    tweetsDf["sentimentScore"] = tweetsDf["cleanedContent"].apply(perform_sentiment_analysis)
    averageScores = tweetsDf.groupby("date")["sentimentScore"].mean().reset_index()
    averageScores = averageScores.rename({"sentimentScore":"averageSentimentScore"}, axis=1)
    return averageScores

def price_chart(pricesDf):
    fig = plt.figure(figsize=(16, 9))
    sns.lineplot(x="openDate", y="close", data=pricesDf)
    plt.xlabel("Date", fontdict={"fontsize":20}, labelpad=20)
    plt.ylabel("Close Price($)", fontdict={"fontsize":20}, labelpad=20)
    st.pyplot(fig)

def predict_test(modelNp, testDfNp):
    # Predict the test data
    future = modelNp.make_future_dataframe(df=testDfNp, n_historic_predictions = len(testDfNp))
    pred = modelNp.predict(future)

    # Rearrange the yhats to match the actual values
    num = 1

    pred = pred.iloc[numLags:-forecast]
    pred.reset_index(inplace=True)

    arrangedDf = pd.DataFrame()
    arrangedDf["date"] = pred["ds"]
    arrangedDf["actualPrice"] = pred["y"]

    for i in range(len(pred)):
        if (num > 3):
            num = 1
        arrangedDf.loc[i, "predictedPrice"] = pred.loc[i, "yhat" + str(num)]
        num += 1
        
    arrangedDf = arrangedDf.set_index("date")

    # Display the errors
    MAE = mean_absolute_error(arrangedDf["actualPrice"], arrangedDf["predictedPrice"])
    MSE = mean_squared_error(arrangedDf["actualPrice"], arrangedDf["predictedPrice"])
    RMSE = math.sqrt(mean_squared_error(arrangedDf["actualPrice"], arrangedDf["predictedPrice"]))
    
    return arrangedDf, MAE, MSE, RMSE

def predict_next(modelNp, testDfNp, forecast):
    # Predict the next 3 days
    future = modelNp.make_future_dataframe(testDfNp, periods=forecast)
    forecast = modelNp.predict(future)

    day1 = forecast.iloc[-3]["yhat1"]
    day2 = forecast.iloc[-2]["yhat2"]
    day3 = forecast.iloc[-1]["yhat3"]
    day1, day2, day3 = round(math.exp(day1),2), round(math.exp(day2),2), round(math.exp(day3),2)

    return day1, day2, day3

def scrape_tweets(query, length):
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(tweets) == length:
            break
        else:
            if ("update" not in tweet.rawContent.lower() and 
                "give" not in tweet.rawContent.lower() and 
                "price" not in tweet.rawContent.lower() and
                "block" not in tweet.rawContent.lower() and
                "bitcoin" in tweet.rawContent.lower()):
                tweets.append({"date": tweet.date, "content": tweet.rawContent})
    return tweets

@st.cache_data
def show_sentiment_vis_and_pred_price(tweetsDf, _modelNp, testDfNp, forecast, pricesDf):
        # Get the dates for actual
        date = pricesDf["openDate"].iloc[-1]
        date1 = date - relativedelta(days=6)
        actualDf = pricesDf[pricesDf['openDate'] >= date1]

        # Get predictions
        day1, day2, day3 = predict_next(_modelNp, testDfNp, forecast)
        # Get the dates for predictions
        date1 = date + relativedelta(days=1)
        date2 = date + relativedelta(days=2)
        date3 = date + relativedelta(days=3)
        predictedDf = pd.DataFrame({"date":[date1, date2, date3], "predictedPrice":[day1, day2, day3]})

        # Add the last actual price to the predictedDf dataframe
        lastActualPrice = actualDf["close"].iloc[-1]
        lastActualDate = actualDf["openDate"].iloc[-1]
        newDf = pd.DataFrame({"date": [lastActualDate], "predictedPrice": [lastActualPrice]})
        concatenatedDf = pd.concat([predictedDf, newDf], ignore_index=True)

        # Plot the predicted prices
        st.subheader("Predicted Close Price Chart")
        fig = plt.figure(figsize=(16, 9))
        sns.lineplot(x="openDate", y="close", data=actualDf, label="Actual")
        sns.lineplot(x="date", y="predictedPrice", data=concatenatedDf, label="Predicted")
        plt.xlabel("Date", fontdict={"fontsize":20}, labelpad=20)
        plt.ylabel("Close Price($)", fontdict={"fontsize":20}, labelpad=20)
        st.pyplot(fig)

        st.subheader("")

        with st.container():
            col1, col2 = st.columns([4, 4])
            with col1:
                # Show the predicted prices
                df = pd.DataFrame({"Date":[date1, date2, date3], "Predicted Close Price":["$"+f"{day1:,}", "$"+f"{day2:,}", "$"+f"{day3:,}"]})
                st.subheader("Predicted Close Prices")
                st.table(df)
            with col2:
                # Show the sentiment for the past 3 days
                st.subheader("Sentiment (Past 3 Days)")
                generate_barplot_sentiment(tweetsDf)
                st.caption("A total of 150 tweets are collected each day.")
        st.subheader("")

        # Show the wordcloud
        st.subheader("Top Words from Tweets in the Past 3 Days")
        generate_wordcloud(tweetsDf)

@st.cache_data
def show_model_info(metrics, _modelNp, testDfNp, ar_layers, learning_rate):
    # Show the hyperparameters of the model
    st.subheader("Hyperparameters of Neural Prophet")
    st.write("The hyperparameters of the Neural Prophet model are as follows:")
    tableHyper = pd.DataFrame({"Hyperparameter":["n_forecasts", "n_lags", "ar_layers", "loss_func", "epochs", "learning_rate"], 
                              "Value":[forecast, numLags, ar_layers, "MSE", 150, learning_rate]})
    st.table(tableHyper)

    # Show the loss plot
    st.subheader("Loss Plot")
    lossPlot = generate_loss_plot(metrics)
    st.pyplot(lossPlot)
    st.subheader(" ")

    # Show the error metrics
    arrangedDf, MAE, MSE, RMSE = predict_test(_modelNp, testDfNp)
    st.subheader("Error Metrics")
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("MAE")
            st.write(f"{MAE:.6f}")
        with col2:
            st.subheader("MSE")
            st.write(f"{MSE:.6f}")
        with col3:
            st.subheader("RMSE")
            st.write(f"{RMSE:.6f}")
    st.subheader(" ")    

    # Show the actual vs predicted plot
    arrangedDf["predictedPrice"] = arrangedDf["predictedPrice"].apply(lambda x: math.exp(x))
    arrangedDf["actualPrice"] = arrangedDf["actualPrice"].apply(lambda x: math.exp(x))
    MAE = mean_absolute_error(arrangedDf["actualPrice"], arrangedDf["predictedPrice"])
    MSE = mean_squared_error(arrangedDf["actualPrice"], arrangedDf["predictedPrice"])
    RMSE = math.sqrt(mean_squared_error(arrangedDf["actualPrice"], arrangedDf["predictedPrice"]))
    errorStr = f"MAE={MAE:.3f}\nMSE={MSE:.3f}\nRMSE={RMSE:.3f}"
    st.subheader("Actual vs Predicted Close Price")
    actualPredictedPlot = generate_actual_vs_predicted_plot(errorStr, arrangedDf)
    st.pyplot(actualPredictedPlot)
    st.write("Note that inverse transformation has been applied to the error metrics shown in the plot.")

def show_home_page(pricesDf):
    date = pricesDf["openDate"].iloc[-1]
    # Show the paragraphs for home page
    st.write("<p style='text-align: justify;'> This application makes predictions by incorporating <b>the past three days' sentiment scores and close prices</b> to predict <b>the next three days of close prices.</b> For example, if the current date is the <b>5th,</b> the sentiment scores and close prices will be from the <b>2nd, 3rd, and 4th.</b> The predicted prices will be for the <b>5th, 6th, and 7th.</b></p>", unsafe_allow_html=True)
    st.write("<p style='text-algin: justify;'> In addition, this web application makes new predictions and refreshes the data every <b>11:00 AM (MYT)</b>. If you have any questions, don't hesitate to contact the developer via <b>tp064059@mail.apu.edu.my</b>.</p>", unsafe_allow_html=True)
    st.write("<p style='text-align: justify;'> Note that this application only provides predictions for <b>BTC.</b> However, the developer plans to expand the coverage to include predictions for other cryptocurrencies, <b>such as ETH and BNB.</b></p>", unsafe_allow_html=True)
    st.subheader("")

    # Show the price chart
    st.subheader("Price Chart of BTC")
    price_chart(pricesDf)
    st.subheader("")
    st.subheader("🗓️ : " + str(date) + " (Previous Day)")

    # Show the open, close, and volume
    with st.container():
        col1, col2, col3 = st.columns(3)
        col1.write("Open Price")
        openPrice = pricesDf["open"].iloc[-1]
        col1.subheader("$ " + f"{openPrice:,}")
        col2.write("Close Price")
        closePrice = pricesDf["close"].iloc[-1]
        col2.subheader("$ " + f"{closePrice:,}")
        col3.write("Volume")
        volume = round(float(pricesDf["volume"].iloc[-1]), 2)
        col3.subheader("BTC " + f"{volume:,}")

def update(pricesDf, tweetsDf, date):
    # Tweets update
    lastRow = len(tweetsDf)
    targetDate = tweetsDf["date"][lastRow-1] + relativedelta(days=1)
    currentDate = date + relativedelta(days=-1)
    tweets = perform_queries(currentDate, targetDate)
    newTweetsDf = pd.DataFrame(tweets)
    newTweetsDf["date"] = newTweetsDf["date"].dt.date
    writeTweetsDf = pd.concat([tweetsDf, newTweetsDf]).reset_index(drop=True)
    writeTweetsDf.to_excel(tweetSheet, index=False)
    tweetsDf = writeTweetsDf

    # Price update
    lastRow = len(pricesDf)
    targetDate = pricesDf["openDate"][lastRow-1] + relativedelta(days=1)
    newPricesDf = retrieve_prices(date, targetDate)
    newPricesDf = preprocess_prices(newPricesDf)
    newPricesDf.drop((len(newPricesDf)-1),axis=0,inplace=True)
    writePricesDf = pd.concat([pricesDf, newPricesDf]).reset_index(drop=True)
    writePricesDf.to_excel(priceSheet, index=False)
    pricesDf = writePricesDf
    pricesDf[["open", "close", "volume"]] = pricesDf[["open", "close", "volume"]].astype(float)

    # Merged update
    averageScores = preprocess_and_perform_sentiment_analysis(tweetsDf)
    mergedDf = pd.merge(averageScores, pricesDf["close"], left_index=True, right_index=True)
    rearrangeColumns = ["date","averageSentimentScore","close"]
    mergedDf = mergedDf.reindex(columns=rearrangeColumns)
    mergedDf.to_excel(mergedSheet, index = False)
    mergedDf["close"] = mergedDf["close"].astype(float)

    return pricesDf, tweetsDf, mergedDf     


