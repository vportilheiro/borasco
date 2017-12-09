import data_collection
import pandas 
import datetime 

if __name__ == '__main__': 
    end_date_str = input("input end date year, month, day (seperated by comma): ")
    end_date = end_date_str.split(",")
    end = datetime.date(int(end_date[0]), int(end_date[1]), int(end_date[2]))
    delta = datetime.timedelta(days=int(input("enter amount of days:")))
    df = data_collection.get_SP500_ts(end-delta, end)
    df.to_csv("SP500_{}day_ts.csv".format(delta))
