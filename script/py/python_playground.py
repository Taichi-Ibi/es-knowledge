#!/usr/bin/env python
# coding: utf-8

# ## モジュールの読み込み
# > モジュールを追加する時はモジュール名のアルファベット順に追加してください

# In[ ]:


from bs4 import BeautifulSoup
import collections
from collections import OrderedDict
from copy import deepcopy
import doctest
import functools
import gc
import getpass
import glob
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import pdfminer
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams, LTContainer, LTTextBox, LTTextLine, LTChar
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
import re
import requests
import sys
import urllib


# In[ ]:


## データが確認されているパス
data_path = '../../sample_data'


# ---
# ## シンタックス
# >

# ### 長い文字列を複数行に分けて書く

# In[ ]:


text1 = '長い文字列を複数行に分けて書く。'    '長い文字列を複数行に分けて書く。'
text2 = ('長い文字列を複数行に分けて書く。'
    '長い文字列を複数行に分けて書く。')
print(text1)
print(text2)


# ### 桁数の大きな数字を見やすくする
# > _はPythonでは無視される

# In[ ]:


num1 = 1_000_000
num2 = 500_000
sum = num1 + num2
print(sum)
print(f'{sum:,}')


# ---
# ## 文字にクォーテーションを付けるのが煩わしい時のテクニック
# >

# In[ ]:


# ExcelやWebのテキストを改行区切りでリストに変換する関数
def text_to_list(text, sep='\n'):
    # textの前後の\nを削除し、sep区切りでリストにして、前後の空白を削除
    return [c.strip() for c in text.strip('\n').split(sep) if c!='']

# 引数にはクォーテーションなしのテキストを与える
l1 = text_to_list('''
customer_id
customer_name
''')

# カンマ区切りも使える
l2 = text_to_list('''customer_id, customer_name''', ',')

print(l1)
print(l2)


# ---
# ## for文
# >

# #### breakでfor文を強制終了させる
# > for文の中で可視化するコードを書いた後に、コードを極力いじらずにグラフを微調整するときに便利

# In[ ]:


l1 = [0, 1, 2]
l2 = []

for value in l1:
    l2.append(value)
print(l2)
l2.clear()

## 特定の箇所で抜ける場合
l2 = []
for value in l1:
    l2.append(value)
    if value==1: break
print(l2)
l2.clear()

# 1回で抜ける場合
for value in l1:
    l2.append(value)
    break
print(l2)


# ---
# ## dict型
# > xxx

# ### KeyErrorを回避しつつget()で辞書型から値を取り出す
# > get()を使うとKeyがない場合にNoneが返ってくる

# In[ ]:


item1 = {'name': 'apple', 'price': 150}
print(item1.get('price'))
print(item1.get('discount'))


# ---
# ## set型
# > set型は重複のない要素の集合

# ### set型の挙動を確認する

# In[ ]:


l1 = ['apple', 'grape', 'banana', 'banana']
# 単純にset型にした場合
print(set(l1))
# list()とするとlist型に変換可能
print(list(set(l1)))


# ### 複数のset型を使う

# In[ ]:


l2 = ['lemon', 'apple', 'peach', 'lemon']
# 和集合を取る
print(set(l1) | set(l2))
print(set(l1 + l2))
# 積集合を取る
print(set(l1) & set(l2))
# 重複しない要素を抽出
print(set(l1) ^ set(l2))


# ## list型とset型の組み合わせ

# ### 2つのリストから重複データを保持せずユニークな差分データだけを求める
# > set型を使うとデータの順序が保持されない点に注意

# In[ ]:


l1 = ['大山', '佐藤', '大山', '佐藤', '山下', '平野', '山下', '平野']
l2 = ['大山', '佐藤']
result = list(set(l1) - set(l2))
print(result)


# ### 2つのリストから共通しない部分の差分データに関しては重複データも残して求める
# > 共通して存在するものだけ削除される

# In[ ]:


result = list(filter(lambda x: x not in l2, l1))
print(result)


# ### 2つのリストからすべてのデータで重複データを残して差分データを求める
# > deepcopyを使っているのは、元のl1を上書き編集しないため

# In[ ]:


result = deepcopy(l1)
foo = [result.remove(v) for v in l2] # removeのreturnはNoneなのでfoo自体に意味はない
print(result)


# ---
# ## 関数
# > xxx

# ### help()で関数の仕様を確認する

# In[ ]:


# ちょっとしたことであれば検索するより早い場合がある
help(pd.DataFrame())


# ### 関数の引数はできるだけキーワード引数で呼び出す

# In[ ]:


def calc_vol(surface, height):
    '''面積と高さから体積を計算する関数'''
    vol = surface*height
    return vol

# 初見の人には数字が何を表しているのかわからない
print(calc_vol(400, 2))

# キーワード引数で呼び出すと数字の意味がよくわかる
print(calc_vol(surface=400, height=2))

# キーワード引数を使うと順序を入れ替えられるというメリットもある
print(calc_vol(height=2, surface=400))


# In[ ]:


# 引数の先頭に*,と書くと、移行の引数はキーワード引数でしか渡せなくなる
def calc_vol_kwds(*, surface, height):
    '''面積と高さから体積を計算する関数'''
    vol = surface*height
    return vol


# ### コードのテスト

# In[ ]:


def second_max(values):
    '''リストの2番目に大きい要素を取得する関数
    >を3つ書いた後テストコードを書き、次の行に正しい実行結果を書く
    >>> l = list(range(0,10))
    >>> print(second_max(l))
    8
    '''
    return max([v for v in values if v!=max(values)])

def third_max(values):
    '''リストの3番目に大きい要素を取得する関数
    実装を間違った例
    >>> l = list(range(0,10))
    >>> print(third_max(l))
    7
    '''
    return max([v for v in values if v!=max(values)])

# コードのテスト
doctest.testmod()


# ## デバッグ・リファクタリングのテクニック
# >知っているとコードの追加や修正が早く行える

# ### スクリプトの実行を任意のセルで中断する

# In[ ]:


# raise Exception()


# In[ ]:


## 括弧内に文字列を入れるとメッセージを表示できる
# error_msg = 'スクリプトの実行を中断しました。'
# raise Exception(error_msg)


# ### スクリプトの実行を中断して対話的デバッグモードに移行する

# In[ ]:


## qまたはquitと入力するとデバッグモードを抜けられる
# import pdb; pdb.set_trace()


# ### reprを使ったprintデバッグ

# In[ ]:


a = 1
b = '1'

# ただのprint分だとintかstringか不明
print(f'a = {a}')
print(f'b = {b}')

# reprを使うと整形前の文字列を出力できる
print(f'a = {repr(a)}')
print(f'b = {repr(b)}')


# ---
# ## スクレイピング
# > 短時間に大量のアクセスをするとサーバーに負荷をかける恐れがある。スクレイピングを禁止しているサイトもあるので利用には注意

# ### 標準モジュールのみでスクレイピングする

# In[ ]:


def get_content_by_tag(url, tag_l, tag_r):
    '''
    特定のhtmlタグに挟まれた要素を表示する

    url: 対象とするサイトのURL
    tag_l: 抽出対象とする開始タグ
    tag_r: 抽出対象とする終了タグ
    return: contentを表示する
    '''

    # htmlを取得
    decodeText = get_html(url)
    # タグに挟まれた要素を表示
    content = re.search(f'{tag_l}(.+?){tag_r}', decodeText).group(1)
    return content

def get_html(url):
    # サイトデータをオブジェクトとして取得
    response = urllib.request.urlopen(url)
    # 元の情報に戻す（エンコードされたデータをデコードする）
    decodeText = response.read().decode("utf-8")
    return decodeText

print(get_content_by_tag('https://estyle-inc.jp/', '<meta name="description" content=', '>'))


# ### BeautifulSoupでテーブルデータをスクレイピングする

# In[ ]:


# Webページを取得し解析
load_url = "https://www.football-lab.jp/sapp/season/"
html = requests.get(load_url)
soup = BeautifulSoup(html.content, "html.parser")

#札幌、鹿島、浦和をリスト化
teams = ['sapp', 'kasm', 'uraw']
#2022、2021、2020年をリスト化
years = ['', '?year=2021', '?year=2020']

#resultに格納
result = []
for i in teams:
    for j in years:
      url = f"https://www.football-lab.jp/{i}/match/{j}"
      scdf = pd.read_html(str(url), header=0)
      scdf = pd.DataFrame([i]*len(scdf[0])).join(scdf, lsuffix='0')
      scdf = pd.DataFrame([j]*len(scdf[0])).join(scdf, lsuffix='0')
      result.append(scdf)

df = pd.concat(result, ignore_index=True)
df.head()


# ---
# ## PDFデータからテキスト抽出

# In[ ]:


with open(f'{data_path}/pdf/sample.pdf', 'rb') as fp:
    with StringIO() as outfp: # 出力先をPythonコンソールするためにIOストリームを取得
        rmgr = PDFResourceManager() # PDFResourceManagerオブジェクトの取得
        lprms = LAParams()          # LAParamsオブジェクトの取得
        with TextConverter(rmgr, outfp, laparams=lprms) as device: # TextConverterオブジェクトの取得
            iprtr = PDFPageInterpreter(rmgr, device) # PDFPageInterpreterオブジェクトの取得
            for page in PDFPage.get_pages(fp): # PDFファイルから1ページずつ解析(テキスト抽出)処理する
                iprtr.process_page(page)
            text = outfp.getvalue()  # Pythonコンソールへの出力内容を取得
print(text[:500])

