import gc
from genericpath import exists, isfile
import glob
import os
import pandas as pd
import sqlite3

# 変数定義
db_local = '../../module/db_local.txt' # localのdb管理
db_remote = '../../module/db_remote.txt' # remoteのdb管理
db_name = 'sample_data' # データを配置しているディレクトリ
dir_csv = f'../../{db_name}/csv'
files = glob.glob(f'{dir_csv}/*.csv') # csvファイルのパスを取得
db_path = f'../../{db_name}/db/{db_name}.db' # 作成するdbのパス

tables = []
for f in files: # ファイルごとの処理
    table = f.split('/')[-1].split('.')[-2] # csvファイル名を取得
    tables.append(table) # データ読み込み後に一覧表示するため、テーブル名をリストに追加

# データ型を定義
dtype = {
    'customer_id': str,
    'gender_cd': str,
    'postal_cd': str,
    'application_store_cd': str,
    'status_cd': str,
    'category_major_cd': str,
    'category_medium_cd': str,
    'category_small_cd': str,
    'product_cd': str,
    'store_cd': str,
    'prefecture_cd': str,
    'tel_no': str,
    'postal_cd': str,
    'street': str
}

def check_db():
    '''
    dbがあれば作成する、dbに更新があれば再読み込みする
    :return: None
    '''
    exist = if_db_exist() # dbがなければ作る
    lcl = get_commit_id(db_local) # localのcommit_idを取得
    rmt = get_commit_id(db_remote) # remoteのcommit_idを取得
    latest = if_db_latest(lcl, rmt) # dbが最新か確認

    if all([exist==1, latest==1]): # dbがあって最新版のときは何もしない
        pass
    else: # 片方でも0の場合はdbを更新してtxtを更新する
        create_db(exist)
        rewrite_commit_id(db_local, rmt)
    finish_msg()
    return

def if_db_exist():
    '''
    db_local.txtがなければ作る
    :return: ファイルがあれば1、なければ0
    '''
    if(not os.path.isfile(db_local)): # db_local.txtがない場合
        with open(db_local, 'w') as f:
            f.write('') # 空のファイルを作成
        existance = 0
    else: # すでにある場合は何もしない
        existance = 1
    return existance

def get_commit_id(path):
    '''
    txtファイル最終行の文字列を取得する
    :return: 最終行の文字列
    '''
    with open(path) as f: # ファイルをtxtとして読み込み
        commit_id = f.read().split('\n')[-1]
    return commit_id  

def if_db_latest(a, b):
    return (a == b)*1

def rewrite_commit_id(path, commit_id):
    '''
    txtファイルに文字列を書き込む（上書き）
    :return: None
    '''
    with open(path, 'w') as f: # ファイル書き込み
        f.write(commit_id)
    return

def create_db(exist=0):
    '''
    csvファイルを読み込んでdbを作成する
    :return: None
    '''
    msg='Creating' if exist==0 else 'Re-creating'
    print(f'{msg}: {db_name}.db')
    if(os.path.isfile(db_path)):
        os.remove(db_path) # dbが残っていれば削除
    for table in tables:
        df = pd.read_csv(f'{dir_csv}/{table}.csv', dtype=dtype) # csvファイルを読み込み
        con = sqlite3.connect(db_path, isolation_level=None) # dbに接続
        cur = con.cursor()
        df.to_sql(table, con, if_exists='replace') # dbの中にtableを作成
        cur.close()
        con.close()
        del df # dfをtable化したら不要なので削除
        gc.collect() # メモリ解放
    return

def finish_msg():
    '''
    作成したテーブルの一覧を表示する
    :return: None
    '''
    print(f'Loaded:\n{tables}') # 作成したテーブルを表示
    return