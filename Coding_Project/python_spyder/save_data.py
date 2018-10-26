import requests
import json
import time
import random
from multiprocessing import pool
from multiprocessing.dummy import Pool as ThreadPool


# 下载第一页数据


def get_one_page(url):
    """ 下载当前页数据 """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    return None

# 解析第一页数据


def parse_one_page(html):
    data = json.loads(html)['cmts']
    for item in data:
        yield{
            'comment': item['content'],
            'date': item['time'].split(' ')[0],
            'rate': item['score'],
            'city': item['cityName'],
            'nickname': item['nickName']
        }

# 保存数据到文本文档


def save_to_txt(page):
    # for i in range(1, 1001):

    url = 'http://m.maoyan.com/mmdb/comments/movie/248566.json?_v_=yes&offset=' + str(page)
    html = get_one_page(url)
    print('正在保存第%d页。' % page)
    for item in parse_one_page(html):
        with open('xie_zheng.txt', 'a', encoding='utf-8') as f:
            f.write(
                item['date'] + ',' + item['nickname'] + ',' + item['city'] + ',' + str(item['rate']) + ',' +
                item['comment'] + '\n')
    time.sleep(5 + float(random.randint(1, 100)) / 20)


if __name__ == '__main__':
    pool = ThreadPool(18)
    pool.map(save_to_txt, [num for num in range(1, 1001)])
    pool.close()
    pool.join()
