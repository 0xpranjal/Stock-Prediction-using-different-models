# ARIMA model for Stock Prediction

 ## Intro
To begin my journey into algorithmic trading I have made multiple models in varying complexity ranging from naive predictions to a fully convolutional neural network. The purpose of building all these models is to create a framework to build future models off of and eventually a whole training system. In addition, this was an experiment designed to see if complex models such as deep neural networks outperform simple models.
</br>
</br>
 <span style="display:block;text-align:center">![Spy Chart](/Images/spy_plot.png)</span>


 ## Relevant Files
 1. [SPY.csv](https://github.com/bsamaha/Python-Trading-Robot/blob/master/SPY.csv) - This file contains the pulled data on the SPY ETF from its inception until 8/31/2020
 2. [SPY Time Series Forecasting.ipynb](https://github.com/bsamaha/Python-Trading-Robot/blob/master/Notebooks/1.%20Time%20Series%20Forecasting%20with%20Naive%2C%20Moving%20Averages%2C%20and%20ARIMA.ipynb) - This notebook contains code showing how to update the SPY.csv to present day and also contains a naive model, 5 day moving average, 20 day moving average,an ARIMA model, and a Recurrent Neural Network model.
 3. [Linear Model Forecast](https://github.com/bsamaha/Python-Trading-Robot/blob/master/Notebooks/2.%20Linear_Model_Forecast.ipynb) - This notebook was created in google colab and may need to be loaded into colab to run as may all notebooks succeeding this one. This is a linear model with a single dense neuron.
 4. [Dense Forecasting](https://github.com/bsamaha/Python-Trading-Robot/blob/master/Notebooks/3.%20Dense_Forecast.ipynb) - This was created in colab and is a model of two dense layers containing 10 units each.
 5. [RNN Notebook](https://github.com/bsamaha/Python-Trading-Robot/blob/master/Notebooks/4.%20RNN_seqtovec_seqtoseq.ipynb) - This notebook was created in colab and shows how I ran this model using RNNs.
 6. [LSTM.ipynb](https://github.com/bsamaha/Python-Trading-Robot/blob/master/Notebooks/5.%20LSTM_Model.ipynb) - This model contains a model built using Long Short Term Memory cells in the recurrent neural network . This notebook was built in Google Colab and is intended to be used in Google Colab for GPU purposes.
 7. [Preprocessing with CNN](https://github.com/bsamaha/Python-Trading-Robot/blob/master/Notebooks/6.Preprocess_CNN.ipynb)  - In colab, this notebooks hows how to use a 1D conv net to preprocess data for a RNN.
 8. [Full CNN - WaveNet](https://github.com/bsamaha/Python-Trading-Robot/blob/master/Notebooks/7.%20Full_CNN_Wavenet.ipynb) - This notebooks created in colab shows how to create a full CNN to use for time series analysis using a wavenet architecture.
 4. [Formulas.py](https://github.com/bsamaha/Python-Trading-Robot/blob/master/Notebooks/formulas.py) - This .py file contains a miscellaneous group of functions I thought I would be using throughout this project. This file is imported into only the "Time Series Forecasting.ipynb". In the notebooks authored using Colab the relevant functions are located inside.

## Project Summary
The resulting error from all models built and tested are shown in the bar graph below. As you can see the most simple model, a Naive forecast outperformed many of the complex deep learning algorithms in terms of error. Surprisingly to me, the LSTM with a 30 day rolling window outperformed all models.

<span style="display:block;text-align:center">![Spy Chart](/Images/model_results.png)</span>



### Data
The data was pulled using the yfinance API. The time period of the data is the entire existance of SPY ETF in January of 1993 until today's date of 9/1/2020. To update this data simply uncomment all cells in the "update data" cell and rerun. If enough time has passed you may want to alter the train,test,validate data splits.

This project was a univariate time series focused on predicting the close price of the next day through various methods. The data is in daily time steps. As you can see the data is just shy of 7,000 data points. The graph below shows the entire data range and how it is broken up into the train,validate,test segments.

<span style="display:block;text-align:center">![Spy Chart](/Images/SPY_train_valid_test_plot.png)</span>

### Model Results

#### 1. ***Naive Forecast Model***</br>
Naive models are naive due to the fact they dont actually "predict". The naive model uses the price from day before as it's prediction as tomorrows price. Since there is not a large change from day to day (usually) in the stock market, this model performs really well.


<span style="display:block;text-align:center">![Naive Model](/Images/naive_forecast_plot.png)</span>
This shows a full view of the entire training period. However, since the predicted and the actual prices are so close it is really hard to see the differences here.

<span style="display:block;text-align:center">![Naive Model Zoom](/Images/naive_forecast_plot_zoom.png) </span>
This is a zoomed in view of the same model focusing on only the last 10 data points. Here you can easily see the Forecast values mimic the Actual values with a 1 day lag.

#### 2. ***Moving Average Models***
Simple Moving Averages (SMA) are a way of smoothing out the noise in the data to get a better idea of which way the signal is trending. These are not good predictive models, but I wanted to showcase these models as they are often used in conjunction with other models to generate trading signals. The Naive Forecast is actually the same thing as a 1 day moving average.

There are a variety of moving average types most commonly simple moving averages or exponential. Simple moving averages take the average of the price over a certain span of time while exponential applies a weight factor to the average that decreases over time.
<span style="display:block;text-align:center">
**EMA=Price(t)×k+EMA(y)×(1−k)</br>
*where:*
t=today</br>
y=yesterday</br>
N=number of days in EMA</br>
k=2÷(N+1)**
</span>
</br>

<span style="display:block;text-align:center">![20 SMA](/Images/20_day_ma_plot.png) </span>

Here we can see that the 20 day moving average is not a good predictor but it is indicative of a trend. 20 Days may sound like an arbitrary number, but it is important to remember there are only 5 trading days in a week and not 7. This means 20 days is a full trading month.

<span style="display:block;text-align:center">![5 SMA](/Images/5_SMA.png) </span>

The 5 SMA follows the actualy values much more closely than the 20 SMA as expected. As 20 days is a trading month, 5 days is a full trading week.

#### 3. ***ARIMA Model***
There is a lot of information about the ARIMA model in the notebook so I will not go into too great of detail here. ARIMA stands for AutoRegressive Integrated Moving Average. This means has 3 main inputs.

For more information on ARIMA models check out my blog [Build an ARIMA Model to Predict a Stock’s Price](https://levelup.gitconnected.com/build-an-arima-model-to-predict-a-stocks-price-c9e1e49367d3)

- 1st input(p) uses the dependent relationship between an observation and some number of lagged observations. An example of this would be movie theaters typically sell the most tickets on Fridays so there is a correlation between the spike in ticket sales every 7 days.
- 2nd input (d) stands for the differencing required to get the data to become stationary. Stationary is just a fancy way of saying the mean of the data does not change over time. The difference is simply Day(T) - Day(t-1).
- 3rd input (q) is the size of the moving average window

An important factor in analyzing time series data is breaking down the seasonality, and trend. Here is the of a season-trend decomposition plot of our data.

<span style="display:block;text-align:center">![Trend Decomp](/Images/Season_Trend_Decomposition.png) </span>


The ARIMA model used was a (p=1,d=1,q=1) model as this was the quickest to train and the difference in performance was very minute. For more information how I came to decide on this model please go examine the SPY Time Series Forecasting.ipynb.</br>
<span style="display:block;text-align:center">![ARIMA Prediction](/Images/arima_predictions.png) </span>

This is a zoomed in image of the same model pictured above.
<span style="display:block;text-align:center">![ARIMA Prediction](/Images/arima_predictions_zoom.png) </span>

#### 4. ***Linear and Dense Model using Keras/Tensorflow***

Output = activation(dot(input, kernel) + bias
That looks familiar doesn't it? It looks almost identical to y = mx+b. The dot product is sum of the products in two sequences. Well, if there is only two sequences with a length of 1 then it is just the product of those two numbers. This simplifies down to the all to familiar y = mx + b.

A dense layer is just a regular layer of neurons in a neural network. Each neuron recieves input from all the neurons in the previous layer, thus densely connected. The layer has a weight matrix W, a bias vector b, and the activations of previous layer a. The following is te docstring of class Dense from the keras documentation:

This first model is a linear model using only one dense layer with a single neuron. This creates a linear model.

<span style="display:block;text-align:center">![ARIMA Prediction](/Images/linear_model.png) </span>


This model is a dense model consisting of 2 different layers with 10 neurons each.

<span style="display:block;text-align:center">![ARIMA Prediction](/Images/dense_forecast.png) </span>

#### 5. ***Reccurent neural network***

Using Keras, this model was build using 2 SimpleRNN layers with 100 neurons each. One model was constructed with a sequence to vector frame work and another model was created with a sequence to sequence framework. The RNNs are where our models begin to get much more complicated. Our input features from here on are 3 dimesnional. Those dimensions are batch size, # of time steps, and the # of input features. Since we are only using the closing price our input features = 1, also know as univariate.

<span style="display:block;text-align:center">![RNN Prediction](/Images/rnn_forecast.png) </span>

#### 6. ***LSTM Model***

LSTM stands for Long Short-Term Memory. This means the cell actually has a memory and is therefore much better at remembering patterns and making predictions based on patterns. This was the best performing model I have built in this project. It is interesting how poorly the 20 day window did, but the 30 day window was the best model by far.

<span style="display:block;text-align:center">![LSTM Prediction](/Images/LSTM_20.png) </span>
<span style="display:block;text-align:center">![LSTM Prediction](/Images/lstm_30day_window.png) </span>


#### 7. ***CNN preprocessing for RNN and Full CNN with Wavenet like architecture***

I was surprised at how poorly the CNN preprocessing model performed. It was the worst performing model by far. The model seems to underpredict upward moves and downward. This is because CNNs have a moving average like trait and it clearly shows in this model.
<span style="display:block;text-align:center">![CNN Preprocess](/Images/cnn_preprocess_rnn_model.png) </span>

Finally, the last model I created was a full CNN with a wavenet like architecture. The wavenet like architecture is further explained in the notebook. This model performed relatively well as it had the 2nd lowest MAE. This model seemed to consistently predict a higher price than actual no matter what direction the general trend was moving in.

<span style="display:block;text-align:center">![CNN Preprocess](/Images/full_cnn_wavenet.png) </span>
