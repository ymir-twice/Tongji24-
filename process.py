import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re   # regular expression

data_path = "raw_data.csv"
data = pd.read_csv(data_path)


def statis_counts_by_month(df: pd.DataFrame, title="The number of procurement announcements issued by the Central Committee"):
    """ count all months' record """
    df["date"] = df["date"].apply(lambda x: x.strftime("%Y/%m"))
    result = df.groupby("date").count()
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    ax.plot(result.iloc[:, 1], color='red')
    ax.set_title(title)
    ax.set_xlabel("Year/month")
    ax.set_ylabel("Number of buying")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
    ax.tick_params(rotation=45)
    fig.show()
    input()

def statis_digital(df: pd.DataFrame):
    """ count all digital procurement """
    pattern = r"数据|数字"
    mask = [re.search(pattern, string) for string in df["title"]]
    mask = list(map(lambda x: False if x == None else True, mask)) 
    df = df.iloc[mask]
    statis_counts_by_month(df.copy(), title="The number of procurement about digital products announcements")    

def statis_ratio(df:pd.DataFrame):
    """ Demonstrate the proportion of digital procurement in total  """ 
    pattern = r"数据|数字"
    mask = [re.search(pattern, string) for string in df["title"]]
    mask = list(map(lambda x: False if x == None else True, mask))
    df["date"] = df["date"].apply(lambda x: x.strftime("%Y"))
    
    df1 = df.iloc[mask]
    df = df.groupby("date").count()
    df1 = df1.groupby("date").count()
    df2 = df1 / df   # ndarray broadcast   

    # corr and fit and t-exam
    x = np.array(range(1, len(df2.index) + 1)).reshape(1, -1)
    x = x.T
    y = np.array(df2["title"]).reshape(1, -1)
    y = y[0]
    print(x)
    print(y)
    input()
    model = LinearRegression()
    model.fit(x, y)
    print("slope:", model.coef_[0])
    print("intercept:", model.intercept_)
    score = model.score(x, y)
    print("score: ", score)

    # picture
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(df2)
    ax.tick_params(rotation=30)
    ax.set_title("the proportion of digital procurement in total")
    fig.show()
    input()



data['date'] = pd.to_datetime(data['date'])
statis_ratio(data)
