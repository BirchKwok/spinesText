import datetime

def date_format(string, format='%Y%m%d'):
    return datetime.datetime.strptime(string, format)

