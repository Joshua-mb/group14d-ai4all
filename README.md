# Sentiment-Based Stock Direction Classifier​

## Background 
- Social media plays an influential role in our day to day lives whether it is in politics, pop culture, or the news. 
- It is through social media posts that we can get an understanding of user’s emotional states or needs on a variety of topics and issues. Companies have used this data in both positive and negative ways in order to predict trends, tailor their products, personalize user experiences, and much more.
- With our project we are looking into how social media posts, more specifically tweets, could be positively leveraged through sentiment analysis in order to help investors make better decisions when it comes to stock trading.

## Project Overview
- Our project investigates whether Twitter sentiment analysis can predict stock price direction for major technology companies.
- We developed machine learning models to perform binary classification (up/down) on daily stock movements for six tech stocks: Apple (AAPL), Google (GOOG), Intel (INTC), Meta (META), Microsoft (MSFT).

## Data
- We selected “Stock Tweets for Sentiment Analysis and Prediction” dataset from Kaggle. The post dates range from 2021 to 2022
- We used a pre-trained VADER model to assign a sentiment value to each stock tweet based on negative, neutral, positive, and compound scores
- The main features we looked into for our project:
  - Stock Price Data: Close Price, Open Price, Volume (1-2 day rolling window)
  - Stock Tweet Data: Sentiment Compound (1-2 day rolling window)
 
### Data Visualizations
- With our focus being on the future direction of stock prices, we want to get an understanding of all of the stock’s performances over time.
- By plotting the stock prices and also looking at the number of shares traded over time for each stock, we can observe trends and patterns and look for any significant drops or rises in each of the stock's prices and from what we interpreted, most of the stocks seemed to be pretty steady over the timeline.
- We also created a visualization to look at the distribution of the sentiment distribution for each stock's dataset of tweets to see if any stock stood out with unusually high positive or negative sentiment compared to the others. From what we observed most of the stocks had a similar distribution in positive, negative, and neutral sentiment. 
<img width="691" height="327" alt="Screenshot 2026-02-10 at 10 29 08 PM" src="https://github.com/user-attachments/assets/7d3aa7dd-d66c-4a98-8644-09aba3488a03" />
<img width="708" height="349" alt="Screenshot 2026-02-10 at 10 29 31 PM" src="https://github.com/user-attachments/assets/10b26706-3b45-4579-8ede-5251ec917ed1" />
<img width="478" height="351" alt="Screenshot 2026-02-10 at 10 29 56 PM" src="https://github.com/user-attachments/assets/8ad85876-d919-4ada-a6d7-91d7e4b5bb6c" />

## Model Selection
- We thought it would be best to use various algorithms to create different models for our data, and then ultimately decide on which one to select based on its accuracy.
- We looked for different algorithms we could use to build a supervised classification model. We wanted to look for algorithms that could perform well with non-linear data since there were not any linear relationships with the features we were looking at. We eventually narrowed down the algorithms to k-nearest neighbors, neural networks, and support vector machine.
  - Support Vector Machine (RBF Kernel): an ML algorithm that maps input data into a high-dimensional feature space and measures the similarity between pairs of data points within the space. The AAPL stock had a test accuracy of 82%, Google had 75%, INTC had 65%, META had 65%, and MSFT had 72%.
  - K-Nearest Neighbors Model: an ML algorithm that works by finding the k closest data points or neighbors to a given input and then makes a prediction based on the majority class. The AAPL stock had 52%, GOOG had 56%, INTC had 46%, META had 50%, and MSFT had 54%.
  - Neural Networks Model: a machine learning algorithm inspired by the human brain that learns patterns through layers of artificial "neurons." The AAPL stock had 84%, GOOG had 81%, INTC had 65%, META had 66%, and MSFT had 86%.
- We ultimately decided on the Neural Networks Model due to the promising accuracies it provided. Neural network models excel at finding subtle patterns in complex data, which is essential when trying to predict stock movements based on multiple factors, including sentiment.

## Sources of Bias in Our Model
- VADER Sentiment Analysis Limitations: VADER was trained on general social media text, not financial language. This means it struggles with finance-specific terminology and context. For example, when someone tweets "this stock is brutally undervalued," VADER might interpret "brutally" as negative, when in a financial context it's actually expressing a positive opportunity. Additionally, VADER cannot reliably detect sarcasm or irony, which is extremely common in financial Twitter. A tweet saying "Great, another brilliant decision by management" might be pure sarcasm, but VADER would read it as positive.
- Twitter Data Collection: Our data source itself is biased. Twitter users skew younger and more tech-savvy than the general investing population. More importantly, we're missing institutional investor sentiment—the hedge funds, mutual funds, and major players who drive most of the actual trading volume in the market.
- Feature Selection Bias: Our model focuses primarily on sentiment and basic stock features like price and volume, but it ignores fundamental factors that professional investors rely on—things like earnings reports, P/E ratios, revenue growth, and interest rate changes.

## Positive and Negative Impacts of our Model
- Positive:
    - Social media is a valuable source on public sentiment due to its day-to-day influence.
    - Sentiment from social media can reveal the emotional state of investors leading to an impact on market momentum.
    - Sentiment data can lead to more informed decisions.
- Negative:
    - Sentiment models may fail to learn context-dependent situations or recognize sarcasm/irony.
    - Social media can be filled with misleading or irrelevant information.
    - Noisy or inconsistent data can reduce predictive accuracy.
 
## Future Steps
- There are two main directions we'd like to pursue: expanding our dataset and building real-time implementation.
    - We'd increase our training data volume since the current dataset is fairly limited—more historical data spanning different market conditions could significantly improve model generalization and accuracy.
We'd also like to extend the time period to capture bull markets, bear markets, crashes, and recovery periods. This would help us understand how sentiment works differently under various market conditions. We would also like to expand this model beyond tech companies and also test on finance, healthcare, and other sectors.
    - We'd build live web scraping to collect social media posts in real-time rather than relying on historical data. This would make the system actually usable for current trading decisions. We'd create an automated prediction system that deploys our model to generate daily predictions based on live sentiment without manual intervention.
And we'd implement streaming data processing to handle continuous Twitter and social media feeds for up-to-the-minute sentiment analysis. These improvements would transform our research project into a practical, deployable tool for investors.


 





 
