# Author
Ilan Meyer Souffir

# btc_price_prediction
This is a long time project which I will constantly modify after each new study on bitcoin price in order to obtain the best results.<br>
We want to know the Bitcoin price in the future at a specific date, in order to achieve this, I will use 2 models : Random Forest and LSTM and modify them with my personal knowledge.

## How to run ?
* In data_loader.py, main.py and main2.py: enter the start date you want to study the btc price until today's date (end date)
* Run main.py : to plot the BTC price graph and all the features we need for our training.
* Run main2.py : to predict the BTC price on the date you want.

## Results for V1 - 16/12/2024
BTC price today : 101 372$
* Predicted price for 31/12/2024 : Random Forest - 86 468$  /  LSTM - 119 442$
* Predicted price for 16/06/2025 : Random Forest - 102 491$  /  LSTM - 133 162$ 
* Predicted price for 16/12/2025 : Random Forest - 102 491$  /  LSTM - 114 607$
* Predicted price for 16/12/2029 : Random Forest - 102 491$  /  LSTM - 154 090$

We can see that we have a problem in the prediction with Random forest, i will improve this model in V2.

## TO DO :
* improve the 2 models to have a better results.
