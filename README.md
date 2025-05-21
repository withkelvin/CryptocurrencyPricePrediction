This application predicts the next 3 days’ Bitcoin close prices by training a NeuralProphet model on the past 3 days’ sentiment scores (scraped via snscrape) and close‑price history. Because Twitter began requiring login in mid‑April 2023—breaking unauthenticated scraping—our sentiment data stopped on April 17 2023, so predictions cover only April 18–20.

To get started, install the dependencies:
```bash
pip install -r requirements.txt
```
Launch the application:
```bash
streamlit run Main.py
```
