

###################################
###################################
###################################
### Dollar Inflation Project ######
### September 18, 2020 ############
### Miguel Sebastian de la Mata ###
###################################
###################################
###################################


####################################
## Summary of project goals ########
####################################

# The purpose of this project is to look at the inflation to the dollar
# caused by COVID-19 related economic fallout, including stimulus and unemployment
# This was inspired by seeing the price of SLV (silver spot price etf) rise about 83%
# from the march COVID-19 crash of 2020 to July of that same year. Gold has followed a similar,
# but not as extreme pattern. Stock prices have also recovered and then gone on to
# reach all-time highs. This was spurred on by Powell saying last week that he wants to
# raise the inflation goal to 2.5% , and seeing him saying that he sees no signs of current inflation.
#
# To me, there seems to be some discrepancy between what the precious metals are saying,
# and what Powell says. To me there is no good reason silver should have doubled in price.
# I think there is inflation happening outside of what Powell sees, and there is no way
# this could have been evened out by job losses when we have the first stimulus, unless
# he is seeing severe repercussions for the lag in a second effective stimulus outside of
# what has been offered by recent executive orders. On top of this I fail to believe that the
# CPI could faithfully represent inflation recently due to supply chain issues,
# and specifically the case where restaurants shutting down has affected how people consume
# the basic items used in the CPI. This is just a gut instinct.
# I would like to use this first data science blog post to either prove myself right or wrong

# import libraries
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
from fredapi import Fred
import seaborn as sns
import requests, bs4
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# This is a required statement for use of FRED API
# "This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis."
# FRED Terms of Use Link: https://research.stlouisfed.org/docs/api/terms_of_use.html
# just going to my FRED API Key at the top so that I don't forget it
fredFile = open(r'C:\Users\sebid\OneDrive\Desktop\fred_API_key.txt', 'r')
fredAPIKey = fredFile.read()
fredFile.close()

# initialize empty data frame to compare our assets
myData = pd.DataFrame()

# This is our list of tickers we want to add to our data frame
# Stock Market:
#       SPY: SPDR S&P 500 ETF Trust. 500 largest market cap weighted US common stocks based
#       on S&P500 index.
#
# Precious Metals:
#       SLV: iShares Silver Trust. A trust holding silver bullion that is generally valued
#           by daily spot price of silver, less fund expenses.
#       GLD: SPDR Gold Shares. A trust holding gold bullion that is generally valued by the
#           daily spot price of gold, less fund expenses.
#       PPLT: Aberdeen Standard Physical Platinum Shares ETF. A trust holding physical
#           platinum that is generally valued by daily spot price of platinum, less fund expenses.
#
#
# Dollar Valuation Assets:
#       DX-Y.NYB: US Dollar/USDX - Index - Cash. Weighted index of currencies used to measure the value
#           of the dollar as compared to currencies of major US trading partners listed in the
#           FOREX market. Market basket of currencies consists of the Euro, Swiss Franc, Japanese Yen,
#           Canadian dollar, British pound, and Swedish Krona, with largest weighting towards the
#           Euro and the least weighting given to the Swiss Franc.
#       TLT: iShares 20+ Year Treasury Bond ETF. An ETF that tracks the performance of US government
#           bonds with 20 or more years until maturity. Lower Fed interest rates tend to drive up the
#           price of the bonds.
#       ^TNX: Treasury Yield 10 Years. An index that tracks the value of 10 year Treasury Notes.
#       CPI: Seasonally adjusted Consumer Price Index based on a market basket of consumer goods.
tickerList = ['SLV', 'GLD', 'PPLT','DX-Y.NYB', 'TLT', '^TNX', 'SPY']

# this is a for loop that iterates through the tickers in
# tickerList, fetches their historical data for closing prices,
# renames the 'Close' column to reflect the ticker name,
# and full joins it to our empty data frame
for ticker in tickerList:
    tickerRetrieve = yf.Ticker(ticker)
    tickerDf = pd.DataFrame(tickerRetrieve.history(period='max')['Close'])
    tickerDf.rename(columns={'Close': str(ticker) + '_Close'}, inplace=True)
    myData = pd.merge(myData, tickerDf, how='outer', right_index=True, left_index=True)

# make date into a column from index
myData.reset_index(inplace=True)

# lets get some FRED CPI data to compare our inflation to
fred = Fred(api_key=fredAPIKey)
CPIdata = fred.get_series('CPIAUCSL')

############################
### Data Cleaning ##########
############################

# assure Date is a datetime object and make sure we don't have any future dates
myData['Date'] = myData['Date'].dt.date
today = dt.date.today()
myData['Date'] = myData[myData['Date'] <= today]
assert myData['Date'].max() <= today
myData['Date'] = myData['Date'].astype('datetime64')

# assure all underlyings are floats
for ticker in tickerList:
    myData[str(ticker) + '_Close'] = myData[str(ticker) + '_Close'].astype('float')
    assert myData[str(ticker) + '_Close'].dtype == 'float'


#############################
### Descriptive Analytics ###
#############################


# lineplot of SLV prices
# we are also going to annotate the worst stock crash of the 2008 financial crisis (Sep 29th, 2008)
# as well as the worst stock crash for 2020 (March 3rd 2020)
ax = sns.lineplot(x='Date', y='SLV_Close', data=myData)
ax.set_title("Price of SLV over time")
ax.set_ylabel("Closing price of SLV (in US dollars)")
plt.axvline(dt.datetime(2020, 3, 12), color='black', linewidth=1)
plt.annotate(text='Black Thursday, 2020',
             xy=(dt.datetime(2020, 3, 12), 20),
             xytext=(dt.datetime(2015, 1, 1), 45),
             color='red',
             arrowprops=dict(arrowstyle='->', color='red', linewidth=2))
plt.axvline(dt.datetime(2008, 9, 29), color='black', linewidth=1)
plt.annotate(text='Worst Stock Crash of 2008',
             xy=(dt.datetime(2008, 9, 29), 8),
             xytext=(dt.datetime(2012, 1, 1), 10),
             color='red',
             arrowprops=dict(arrowstyle='->', color='red', linewidth=2))
plt.show()

# melt data for sns lineplots
myDataMelted = pd.melt(myData, 'Date', var_name='Underlying', value_name='Price')

# focus in on 2008 to present
myData2008Forward = myData.loc[pd.to_datetime(myData['Date'])
                               >= dt.datetime.strptime('2008-01-01', '%Y-%m-%d')]

# comparing all underlyings in a lineplot from 2008
myData2008ForwardMelted = pd.melt(myData2008Forward, 'Date', var_name='Underlying', value_name='Price')
ax = sns.lineplot(x='Date', y='Price', hue='Underlying', data=myData2008ForwardMelted)
ax.set_title('Historic prices of underlyings from 2008 onward')
plt.axvline(dt.datetime(2020, 3, 12), color='black', linewidth=1)
plt.annotate(text='Black Thursday, 2020',
             xy=(dt.datetime(2020, 3, 12), 20),
             xytext=(dt.datetime(2015, 1, 1), 45),
             color='red',
             arrowprops=dict(arrowstyle='->', color='red', linewidth=2))
plt.axvline(dt.datetime(2008, 9, 29), color='black', linewidth=1)
plt.annotate(text='Worst Stock Crash of 2008',
             xy=(dt.datetime(2008, 9, 29), 225),
             xytext=(dt.datetime(2010, 1, 1), 225),
             color='red',
             arrowprops=dict(arrowstyle='->', color='red', linewidth=2))
plt.show()

# focus in on just 2020
myData2020Forward = myData.loc[pd.to_datetime(myData['Date'])
                               >= dt.datetime.strptime('2020-01-01', '%Y-%m-%d')]

# comparing all underlyings in a lineplot from 2020
myData2020ForwardMelted = pd.melt(myData2020Forward, 'Date', var_name='Underlying', value_name='Price')
ax = sns.lineplot(x='Date', y='Price', hue='Underlying', data=myData2020ForwardMelted)
ax.set_title('Historic prices of underlyings from 2020 onward')
plt.axvline(dt.datetime(2020, 3, 12), color='black', linewidth=1)
plt.annotate(text='Black Thursday, 2020',
             xy=(dt.datetime(2020, 3, 12), 20),
             xytext=(dt.datetime(2020, 1, 1), 45),
             arrowprops=dict(arrowstyle='->', color='red', linewidth=2))
plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
plt.show()

# these lineplots don't really scale well, lets look at weekly percent change and compare them
# let's use wednesday since fridays and mondays may be holidays sometimes
myDataPctChange = myData2008Forward[myData2008Forward['Date'].dt.weekday == 2]
myDataPctChange.set_index('Date', inplace=True)
myDataPctChange = myDataPctChange.pct_change().dropna() * 100
myDataPctChange.reset_index(inplace=True)

# lets melt this data so we can compare them in beeswarm plots
myDataPctChangeMelted = pd.melt(myDataPctChange, 'Date',
                                           var_name='Underlying', value_name='PctChange')

# comparing all underlyings daily percent change in a bee swarmplot
ax = sns.swarmplot(x='Underlying', y='PctChange', data=myDataPctChangeMelted)
ax.set_title("Historic weekly percent change in prices of underlyings, 2008 forward")
ax.set_ylabel("Weekly percent change in price")
plt.show()


# now let's look at percent change from 2020 onwards, performing the same melting as we did
# before so that we can put it in a swarmplot
myDataPctChange2020Forward = myDataPctChange.loc[pd.to_datetime(myDataPctChange['Date'])
                                                 >= dt.datetime.strptime('2020-01-01', '%Y-%m-%d')]
myDataPctChangeMelted2020Forward = pd.melt(myDataPctChange2020Forward, 'Date',
                                           var_name='Underlying', value_name='PctChange')
print(myDataPctChange2020Forward.head())

# comparing all underlyings daily percent change in a bee swarmplot
ax = sns.swarmplot(x='Underlying', y='PctChange', data=myDataPctChangeMelted2020Forward)
ax.set_title("Historic weekly percent change in prices of underlyings from 2020 forward")
ax.set_ylabel("Weekly percent change in price")
plt.show()

# heatmap to examine correlations of underlyings
# we will separate the data from 2008-2020 and the data from 2020 onward
# this will actually be 2010 as PPLT didnt start until 2010 and we knocked out all na values
myData2008ForwardDropNA = myData2008Forward.drop(columns='Date').dropna()
myData2008ForwardDropNA = myData2008ForwardDropNA.loc[pd.to_datetime(myData['Date'])
                                                      < dt.datetime.strptime('2020-01-01', '%Y-%m-%d')]
ax = sns.heatmap(myData2008ForwardDropNA.corr(),
                 annot=True,
                 cmap='PiYG',
                 square=True,
                 mask=np.triu(np.ones_like(myData2008ForwardDropNA.corr(), dtype=np.bool)),
                 vmin=-1,
                 vmax=1,
                 center=0,
                 cbar_kws={"shrink": .5})
ax.set_title('Correlation between underlyings from 2010 to 2020')
plt.xticks(rotation=30)
ax.tick_params(left=False, bottom=False)
for tick in ax.xaxis.get_majorticklabels():
    tick.set_horizontalalignment("right")
plt.show()

# let's do a heatmap of just the year 2020
# and see if there are any notable differences
myData2020ForwardDropNA = myData2020Forward.drop(columns='Date')
ax = sns.heatmap(myData2020ForwardDropNA.corr(),
                 annot=True,
                 cmap='PiYG',
                 square=True,
                 mask=np.triu(np.ones_like(myData2020ForwardDropNA.corr(), dtype=np.bool)),
                 vmin=-1,
                 vmax=1,
                 center=0,
                 cbar_kws={"shrink": .5})
ax.set_title('Correlation between underlyings from 2020 onward')
plt.xticks(rotation=30)
ax.tick_params(left=False, bottom=False)
for tick in ax.xaxis.get_majorticklabels():
    tick.set_horizontalalignment("right")
plt.show()

# let's go month by month and facet them
myData2020Forward['Month'] = pd.to_datetime(myData2020Forward['Date']).dt.month
myData2020ForwardDropNA = myData2020Forward.drop(columns='Date')
myData2020ForwardDropNA = myData2020ForwardDropNA[myData2020ForwardDropNA['Month'] < 9]
# gotta do some tricky shit to facet heatmaps
g = sns.FacetGrid(myData2020ForwardDropNA, col='Month', col_wrap=4)
g.map_dataframe(lambda data,
                       color: sns.heatmap(data.drop('Month', axis=1).corr().round(decimals=2),
                                          linewidths=0,
                                          annot=True,
                                          square=True,
                                          cbar=False,
                                          vmax=1,
                                          vmin=-1,
                                          center=0,
                                          cmap='PiYG',
                                          mask=np.triu(np.ones_like(data.drop('Month', axis=1).corr(),
                                                                    dtype=np.bool))).tick_params(left=False,
                                                                                                 bottom=False))
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Correlation between underlyings for each month of 2020")
g.fig.subplots_adjust(hspace=.3, wspace=-0.05)
g.set_xticklabels(['SLV_Close', 'GLD_Close', 'PPLT_Close','DX-Y.NYB_Close', 'TLT_Close', '^TNX_Close', ''])
g.set_yticklabels(['', 'GLD_Close', 'PPLT_Close','DX-Y.NYB_Close', 'TLT_Close', '^TNX_Close', 'SPY'])
plt.show()



# Now we'll start looking at CPI in comparison
# convert CPI series to dataframe
CPIdata = pd.Series(CPIdata).to_frame(name='CPI')

# the CPI data is monthly, so we need to interpolate to fill every day
CPIdata = CPIdata['CPI'].resample('D').interpolate(method='time')

# now we can add it to our data frame, but need to set date on our index for myData first
myData.set_index('Date', inplace=True)
myData = pd.merge(myData, CPIdata, how='outer', right_index=True, left_index=True)

# next we will construct a precious metals market basket
# we will do this by taking the sum of the metals market caps and applying the
# percentage of each to its respective weighting

# yahoo finance isn't working for market cap data stuff, lets try some beautiful soup stuff

# our web scraping function
def getMarketCap(ticker):
    # build the site address from the ticker
    site = 'https://finance.yahoo.com/quote/' + ticker + '/'

    # request the page and save it as a response object
    res = requests.get(site)

    # check the http status of our get request before continuing
    res.raise_for_status()

    # scrape the text from our response object and save it as soup
    # the features option keeps a warning bug from popping up
    # its not dangerous but it looks ugly
    soup = bs4.BeautifulSoup(res.text, features="lxml")

    # select our CSS traceback element to display the price of the product
    # and save this page element as elems
    elems = soup.select("#quote-summary > div.D\(ib\).W\(1\/2\).Bxz\(bb\).Pstart\(12px\).Va\(t\).ie-7_D\(i\).ie-7_Pos\(a\).smartphone_D\(b\).smartphone_W\(100\%\).smartphone_Pstart\(0px\).smartphone_BdB.smartphone_Bdc\(\$seperatorColor\) > table > tbody > tr:nth-child(1) > td.Ta\(end\).Fw\(600\).Lh\(14px\)")

    # return index 0 from the elems (the price)s
    return(elems[0].text.strip())

# metals we will use in our metals basket
metalList = ['SLV', 'GLD', 'PPLT']

# initialize market cap data frame
basketDict = {}

# get market cap for every day for metals basket
# the web scraping returns strings like '5.5B',
# so we also need to convert it to a float and multiply it by the letter classifier
for metal in metalList:
    if getMarketCap(metal).find('M') != -1:
        basketDict[metal] = round(float(getMarketCap(metal).replace('M', '')) * 1000000)
    elif getMarketCap(metal).find('B') != -1:
        basketDict[metal] = round(float(getMarketCap(metal).replace('B', '')) * 1000000000)
    else:
        basketDict[metal] = getMarketCap(metal)
        print('Could not convert string to int for ' + metal)

print(basketDict)

# now that we have the market cap values as numbers, we want to find
# our percent of total sum so we can apply this weight to our metals
# in our weighted basket
marketCapTotal = sum(basketDict.values())
for key, value in basketDict.items():
    basketDict[key] = value / marketCapTotal

print(basketDict)

# let's make a column in our myData data frame for our new market basket
myData['metalsMarketBasket_Close'] = round((myData['SLV_Close'] * basketDict['SLV']) + \
                                   (myData['GLD_Close'] * basketDict['GLD']) + \
                                   (myData['PPLT_Close'] * basketDict['PPLT']), 2)

# lets make a new data frame with just our CPI and market basket to do some A/B testing
inflationTestDF = myData.dropna()
inflationTestDF = inflationTestDF[['CPI', 'metalsMarketBasket_Close']]

# now we will split this data frame into two groups for our A/B testing: before 2020 and after
inflationTestDFBefore2020 = inflationTestDF[inflationTestDF.index < dt.datetime(2020, 1, 1)]
inflationTestDF2020 = inflationTestDF[inflationTestDF.index >= dt.datetime(2020, 1, 1)]

# first we will take the correlation between our metal basket price and CPI for 2020
corrInflation2020 = inflationTestDF2020['metalsMarketBasket_Close']\
    .corr(inflationTestDF2020['CPI'])
print(corrInflation2020)

# next we will take the correlation between our metal basket price and CPI before 2020
corrInflationBefore2020 = inflationTestDFBefore2020['metalsMarketBasket_Close']\
    .corr(inflationTestDFBefore2020['CPI'])
print(corrInflationBefore2020)

# here we will perform a pairs bootstrap our samples with sample to determine if the
# difference in our correlations is significant

# first I am making a function that calculates correlation.
# this sounds super dumb, but if I just use np.corrcoef()
# I get a message saying:
# ValueError: setting an array element with a sequence.
# I couldn't figure it out, but this works, so let's just be happy for that
def pearson_r(x, y):
    return (
        np.sum((x - np.mean(x)) * (y - np.mean(y)))
        / np.std(x)
        / np.std(y)
        / np.sqrt(len(x))
        / np.sqrt(len(y))
    )

# first we will  create a function to draw random samples from a 1-D array
def draw_bs_sample(data):
    return np.random.choice(data, size=len(data))

# next we will use the previous function to draw a sample at random, but in pairs
def draw_bs_pairs(x, y):
    inds = np.arange(len(x))
    bs_inds = draw_bs_sample(inds)
    return x[bs_inds], y[bs_inds]

# next we will use this function to draw correlations on our random bootstrapped pairs
def draw_bs_pairs_reps_pearson(x, y, size=1):
    bs_reps_corr = np.empty(size)
    for i in range(size):
        bs_reps_corr[i] = pearson_r(*draw_bs_pairs(x, y))
    return bs_reps_corr

# let's set our number of replicates as a variable to make sure they are the same for both
nreps = 100000

# now lets put all this into action and draw replicates on both before and after 2020
bsReps2020 = draw_bs_pairs_reps_pearson(inflationTestDF2020['metalsMarketBasket_Close'].values,
                                 inflationTestDF2020['CPI'].values, size=nreps)

bsRepsBefore2020 = draw_bs_pairs_reps_pearson(inflationTestDFBefore2020['metalsMarketBasket_Close'].values,
                                 inflationTestDFBefore2020['CPI'].values, size=nreps)

# let's calculate our confidence interval for 2020
bsReps2020CI = np.percentile(bsReps2020, [2.5, 97.5])

# Make a histogram of the results for 2020 replicates,
# showing empirical correlation and confidence intervals
ax = plt.hist(bsReps2020, bins=50, density=True, stacked=True)
ax = plt.xlabel('Correlation between metals basket and CPI 2020')
ax = plt.ylabel('PDF')
plt.axvline(corrInflation2020, color='red', linewidth=1)
plt.axvline(bsReps2020CI[0], color='black', linewidth=1)
plt.axvline(bsReps2020CI[1], color='black', linewidth=1)
plt.show()

# let's repeat this for before 2020
bsRepsBefore2020CI = np.percentile(bsRepsBefore2020, [2.5, 97.5])
# Make a histogram of the results for 2020 replicates,
# showing empirical correlation and confidence intervals
ax = plt.hist(bsRepsBefore2020, bins=50, density=True, stacked=True)
ax = plt.xlabel('Correlation between metals basket and CPI before 2020')
ax = plt.ylabel('PDF')
plt.axvline(corrInflationBefore2020, color='red', linewidth=1)
plt.axvline(bsRepsBefore2020CI[0], color='black', linewidth=1)
plt.axvline(bsRepsBefore2020CI[1], color='black', linewidth=1)
plt.show()

# Now let's focus on the question at hand here:
# Is the difference in correlation between these two periods statistically significant?
# Our null hypothesis would assume that the bootstrapped data would be the same as the empirical
# so let's calculate our p-value: what percent of the data contains a difference in correlation
# at least as high as what we observed in our empirical difference?

# first we will take the difference in correlation for these two periods in the empirical data
empiricalCorrelationDiff = corrInflation2020 - corrInflationBefore2020
print(empiricalCorrelationDiff)

# next we will create an array of the differences in correlation for our bootstrapped pairs
bsCorrelationDiff = bsReps2020 - bsRepsBefore2020

# now we calculate our p-value
p = np.sum(bsCorrelationDiff >= empiricalCorrelationDiff) / len(bsCorrelationDiff)
print(p)

# let's plot it for good measure and include the p-value in the title
# showing empirical correlation and confidence intervals
ax = plt.hist(bsCorrelationDiff, bins=50, density=True, stacked=True)
ax = plt.xlabel('Difference in Correlation between metals basket and \nCPI before '
                'and after 2020 (p-value=' + str(p) + ' of bootstrapped\n samples with '
                                                      'at least this high of a difference in correlation)')
ax = plt.ylabel('PDF')
plt.axvline(empiricalCorrelationDiff, color='red', linewidth=1)
plt.show()

# looks like we have a basis to build a model to construct where CPI
# 'should be' based on other underlyings
# first lets get our data ready
myData.drop(columns='metalsMarketBasket_Close', inplace=True)

# let's also pop Date out as a column so we can account for time based autocorrelation
myData.reset_index(inplace=True)
myData.rename(columns={'index':'Date'}, inplace=True)
myData.dropna(inplace=True)
myData.info()

# first lets split this into pre 2020 and post 2020 because
# we will be predicting on our 2020 CPI values
# focus in on just 2020
myData2020Forward = myData.loc[pd.to_datetime(myData['Date'])
                               >= dt.datetime.strptime('2020-01-01', '%Y-%m-%d')]

# focus in on just 2020
myDataBefore2020 = myData.loc[pd.to_datetime(myData['Date'])
                               < dt.datetime.strptime('2020-01-01', '%Y-%m-%d')]

# in order for our regression model to work, we must convert our dates into numbers
# we could have done this before, but it is easier to do it after splitting the data by date
myDataBefore2020['Date'] = myDataBefore2020['Date'].map(dt.datetime.toordinal)
myData2020Forward['Date'] = myData2020Forward['Date'].map(dt.datetime.toordinal)

# next we want to split our dependent variable (CPI)
# off from our independent variables
y2020Forward = myData2020Forward['CPI']
yBefore2020 = myDataBefore2020['CPI']

x2020Forward = myData2020Forward.drop(columns='CPI')
xBefore2020 = myDataBefore2020.drop(columns='CPI')

# now want to split our data before 2020 for both dependent and
# independent variables into train and test sets
# we will split 30% of the data off as our test set and leave 70% for training
xTrain, xTest, yTrain, yTest = train_test_split(xBefore2020, yBefore2020,
                                                test_size=0.3,
                                                random_state=42)


# we will use an Elastic Net to penalize our variable coefficients as a compromise
# between the penalties used in LASSO and Ridge variable weighting


# first we will want to create a space in which the hyperparameter L1 can be tuned
l1Space = np.linspace(0, 1, 100)
alphaSpace = np.logspace(-5,2,8)
paramGrid = {'alpha' : alphaSpace, 'l1_ratio' : l1Space}

# next we will instantiate our elastic net
elasticNet = ElasticNet()

# then we will set up our GridSearchCV object with 5-fold cross-validation
gridCV = GridSearchCV(elasticNet, paramGrid, cv=5)

# now we will fit our GridSearchCV object to our training data
gridCV.fit(xTrain, yTrain)

# now we will use our model to predict on our test data and evaluate it's metrics
yPredict = gridCV.predict(xTest)

# compute our r2 four our model
r2 = gridCV.score(xTest, yTest)

# compute mean squared error for our model
mse = mean_squared_error(yTest, yPredict)

# print out our metrics
# note: our l1 ratio calculated to 0, essentially meaning
# that the best model was derived using ridge regression
print('Tuned Elastic Net Alpha and L1 Ratio: {}'.format(gridCV.best_params_))
print('Tuned Elastic Net R-squared: {}'.format(r2))
print('Tuned Elastic Net Mean Squared Error (MSE): {}'.format(mse))

# now let's rebuild our elastic net with retuned
# hyperparameters so we can look at our feature importance
model = ElasticNet(alpha=gridCV.best_params_['alpha'], l1_ratio=gridCV.best_params_['l1_ratio'])
model.fit(xTrain, yTrain)
elasticNetCoef = model.coef_

# create an index object of our column names to plot our feature importance with
xCols = xTrain.columns

# now let's plot our coefficients
plt.style.use('ggplot')
plt.plot(range(len(xCols)), elasticNetCoef)
plt.xticks(range(len(xCols)), xCols.values)
plt.title("Feature importance of independent variables for\nElasticNet model (coefficient values)")
plt.margins(0.02)
plt.show()

# now let's do our 2020 predictions of CPI based on this model
predict2020CPI = model.predict(x2020Forward)

# reset our myData2020Forward index to zero so we can attach these predicted CPIs
myData2020Forward.reset_index(drop=True, inplace=True)

# now let's add this back into our 2020 dataframe
myData2020Forward = pd.concat([myData2020Forward, pd.DataFrame(predict2020CPI, columns=['predicted_CPI'])],
                              axis=1)

# next we will change the dates from ordinal back to dates so we can union them back together
myData2020Forward['Date'] = myData2020Forward.iloc[:, 0].astype(int).map(dt.date.fromordinal)
myDataBefore2020['Date'] = myDataBefore2020.iloc[:, 0].astype(int).map(dt.date.fromordinal)

# now we set the dates as indexes for the union
myDataBefore2020.set_index('Date', inplace=True)
myData2020Forward.set_index('Date', inplace=True)

# now let's union them
myNewData = myDataBefore2020.append(myData2020Forward, sort=True)

# now let's make a lineplot showing the difference between reported and predicted CPI for 2020
plt.plot(myData2020Forward.index, myData2020Forward['CPI'], color='black', label='Recorded CPI')
plt.plot(myData2020Forward.index, myData2020Forward['predicted_CPI'], color='red', label='Modeled CPI')
plt.title('Recorded CPI vs Modeled CPI for 2020')
plt.legend()
plt.show()

# now we want to extract the most recent CPI, the most recent predicted CPI,
# and the CPI from 1 before the most recent CPI to compare year-over-year inflation
latestPredictedCPI = myNewData.loc[myNewData.index.max(), ['predicted_CPI']].to_numpy()
print(latestPredictedCPI)
latestCPI = myNewData.loc[myNewData.index.max(), ['CPI']].to_numpy()
print(latestCPI)
yearFromLatestCPI = myNewData.loc[(myNewData.index.max() - pd.Timedelta('365 days')), ['CPI']].to_numpy()
print(yearFromLatestCPI)

# first let's calculate year over year recorded inflation
recordedInflation = ((latestCPI - yearFromLatestCPI) / yearFromLatestCPI) * 100
predictedInflation = ((latestPredictedCPI - yearFromLatestCPI) / yearFromLatestCPI) * 100

# print our inflation calculation results
print('Recorded year over year inflation for 7/31/2020: ' + np.array2string(recordedInflation)
      .replace('[', '')
      .replace(']', '') + '%')
print('Modeled year over year inflation for 7/31/2020: ' + np.array2string(predictedInflation)
      .replace('[', '')
      .replace(']', '') + '%')
