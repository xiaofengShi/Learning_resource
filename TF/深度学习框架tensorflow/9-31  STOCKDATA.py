# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 08:53:25 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

import urllib.request
import re
##def downback(a,b,c):
##    ''''
##    a:已经下载的数据块
##    b:数据块的大小
##    c:远程文件的大小
##   '''
##    per = 100.0 * a * b / c
##    if per > 100 :
##        per = 100
##    print('%.2f%%' % per)
stock_CodeUrl = 'http://quote.eastmoney.com/stocklist.html'
#获取股票代码列表
def urlTolist(url):
    allCodeList = []
    html = urllib.request.urlopen(url).read()
    html = html.decode('gbk')
    s = r'<li><a target="_blank" href="http://quote.eastmoney.com/\S\S(.*?).html">'
    pat = re.compile(s)
    code = pat.findall(html)
    for item in code:
        if item[0]=='6' or item[0]=='3' or item[0]=='0':
            allCodeList.append(item)
    return allCodeList
    

start = '20161031'
end='20161231'
'''
allCodelist = urlTolist(stock_CodeUrl)
for code in allCodelist:
    print('正在获取%s股票数据...'%code)
    if code[0]=='6':
        url = 'http://quotes.money.163.com/service/chddata.html?code=0'+code+\
        '&start='+start+'&end='+end+'&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'
    else:
        url = 'http://quotes.money.163.com/service/chddata.html?code=1'+code+\
        '&start='+start+'&end='+end+'&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'
    urllib.request.urlretrieve(url,'d:\\all_stock_data\\'+code+'_'+end+'.csv')#可以加一个参数dowmback显示下载进度
'''    

#test
code='600000'
url = 'http://quotes.money.163.com/service/chddata.html?code=0'+code+\
        '&start='+start+'&end='+end+'&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'
urllib.request.urlretrieve(url,'d:\\all_stock_data\\'+code+'_'+end+'.csv')#可以加一个参数dowmback显示下载进度
    