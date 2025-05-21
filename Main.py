import Functions as f
from streamlit_lottie import st_lottie
import streamlit as st

st.set_page_config(page_title="Cryptocurrency Price Prediction", page_icon="📈")
lottieCoding = f.load_lottie("https://assets5.lottiefiles.com/packages/lf20_GhbJ7XGqaB.json")
now = f.datetime.now()
currentDate = f.datetime.now().date()
ar_layers = [100,100,100,100,100]
learning_rate = 0.01

with st.spinner("Loading data and model..."):
    pricesDf, tweetsDf, mergedDf = f.reload_data()
    trainDfNp, testDfNp = f.preprocess_data_np(mergedDf)
    modelNp, metrics = f.develop_np_model(ar_layers, learning_rate, trainDfNp, testDfNp)

# if tweetsDf.iloc[-1, 0] != (currentDate+f.relativedelta(days=-1)) and now.hour >= 11:
#     st.cache_data.clear()
#     with st.spinner("Updating data and model..."):
#         pricesDf, tweetsDf, mergedDf = f.update(pricesDf, tweetsDf, currentDate)
#         pricesDf, tweetsDf, mergedDf = f.reload_data()
#         trainDfNp, testDfNp = f.preprocess_data_np(mergedDf)
#         modelNp, metrics = f.develop_np_model(ar_layers, learning_rate, trainDfNp, testDfNp)
# else:
#     if tweetsDf.iloc[-1, 0] != (currentDate+f.relativedelta(days=-1)) and (now.hour >= 0 and now.hour < 11):   
#         if tweetsDf.iloc[-1, 0] != (currentDate + f.relativedelta(days=-2)):
#             previousDate = currentDate + f.relativedelta(days=-1)
#             st.cache_data.clear()
#             with st.spinner("Updating data and model..."):
#                 pricesDf, tweetsDf, mergedDf = f.update(pricesDf, tweetsDf, previousDate)
#                 pricesDf, tweetsDf, mergedDf = f.reload_data()
#                 trainDfNp, testDfNp = f.preprocess_data_np(mergedDf)
#                 modelNp, metrics = f.develop_np_model(ar_layers, learning_rate, trainDfNp, testDfNp)

col1, col2 = st.columns([1.5, 4])
with col1:
    st_lottie(lottieCoding, width=158, key="coding")  
with col2:
    st.title(" ")
    st.title("Cryptocurrency Price Prediction")

st.write("<style>hr {margin-top: 0px;}</style>", unsafe_allow_html=True)
st.write("<hr>", unsafe_allow_html=True)
options = ["Home", "About Models", "Predictions"]
st.sidebar.title("Options")
choice = st.sidebar.selectbox("Please select a page", options)

if choice == "Home":
    st.header("Welcome")
    f.show_home_page(pricesDf)
elif choice == "About Models":
    st.header("About Models")
    st.write("<p style='text-align: justify;'> To provide a more comprehensive understanding of each expander, there are several key components that are included. These components consist of <b>the hyperparameters, a loss plot, error metrics, and an actual‑vs‑predicted close‑price plot.</b></p>", unsafe_allow_html=True)
    with st.expander("Model for BTC"):
        f.show_model_info(metrics, modelNp, testDfNp, ar_layers, learning_rate)
elif choice == "Predictions":
    st.header("Predictions") 
    st.write("<p style='text-align: justify;'> To facilitate a better understanding of each expander, there are certain essential components that are included. These components comprise <b>a predicted close‑price chart, a table displaying the predicted close prices, a sentiment bar chart, and a word cloud.</b></p>", unsafe_allow_html=True)
    with st.expander("BTC"): 
        f.show_sentiment_vis_and_pred_price(tweetsDf, modelNp, testDfNp, f.forecast, pricesDf)
           

     
