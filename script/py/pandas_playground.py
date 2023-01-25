#!/usr/bin/env python
# coding: utf-8

# ## モジュールの読み込み
# > モジュールを追加する時はモジュール名のアルファベット順に追加してください

# In[ ]:


import gc
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyperclip
import re
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error


# ## サンプルデータの読み込み
# > ここはいじっちゃダメ

# In[ ]:


# サンプルデータを読み込む
data_path = '../../sample_data'
with open(f'{data_path}/dtype/dtype_str.txt') as f:
    col = f.read().split('\n')
dtype = {'dtype': {c:'str' for c in col}}
customer = pd.read_csv(f'{data_path}/csv/customer.csv', **dtype)
category = pd.read_csv(f'{data_path}/csv/category.csv', **dtype)
product = pd.read_csv(f'{data_path}/csv/product.csv', **dtype)
receipt = pd.read_csv(f'{data_path}/csv/receipt.csv', **dtype)
store = pd.read_csv(f'{data_path}/csv/store.csv', **dtype)
geocode = pd.read_csv(f'{data_path}/csv/geocode.csv', **dtype)


# ---
# ## データの読み込み
# > 大きなデータを扱うときやスクリプトをシステムに組み込むときはオプションを手動で設定しておきたい

# ### 読み込む行数やカラム名を指定
# > 横に長いデータを扱うときやデータの確認をするときに便利

# In[ ]:


# nrows=0とするとカラム名が確認できる
display(pd.read_csv(f'{data_path}/csv/customer.csv', nrows=0))
# カラム名を指定して先頭5行だけ読み込み
usecols = ['customer_id', 'status_cd']
display(pd.read_csv(f'{data_path}/csv/customer.csv', usecols=usecols, nrows=5))


# ### 型指定をして読み込む
# > 返されたエラーによりデータの破損を検知できる  
# > 数値範囲がわかっているのであればint32で読み込むとメモリ節約になる（※後述）  
# > 型推論が実行されないため読み込みが高速化することがある

# In[ ]:


dtype = {'dtype': 
            {
                'sales_ymd': 'int64',
                'sales_epoch': 'int64',
                'store_cd': 'object',
                'receipt_sub_no': 'int64',
                'customer_id': 'object',
                'product_cd': 'object',
                'quantity': 'int64',
                'amount': 'int64'
            }
}
encd = {'encoding': 'utf-8'}
kwds = {**dtype, **encd}

df = pd.read_csv(f'{data_path}/csv/receipt.csv', **kwds)
display(df.head(1))
df.info()


# ---
# ## メモリの節約
# > ローカルPCでも大きなファイルを扱えたり、処理が早くなったりする

# ### int型の数値範囲

# In[ ]:


def calc_int_range(bit):
    return int(2**bit/2)

def calc_total_length(bit):
    num_range = calc_int_range(bit)
    num_len = np.round(int(np.log10(num_range)),0)+1
    comma_cnt = (num_len-1)//3
    total_length = num_len + comma_cnt
    return total_length

for bit in [2**c for c in range(3,7)]:
    num_min = -calc_int_range(bit)
    num_max = -num_min-1
    max_len = calc_total_length(64)+1 # +1はマイナス符号の分
    print(f'int{bit:<2}: {num_min:>{max_len},} ~ +{num_max:<{max_len},}')


# ### データ型の変換によるメモリの節約

# #### int32の活用
# > int16は3万弱までしか扱えないが、int32は20億まで扱える

# In[ ]:


# 100万行のDataFrameを作成。整数のみを含むカラムはデフォルではint64となる
df1 = pd.DataFrame(index=(r:=range(0, 1_000_000)), columns=['value'], data=r)
print(df1.info(), '\n')

# 値の範囲に問題なければint32にすることでメモリ使用量を半減できる
df1['value'] = df1['value'].astype('int32')
print(df1.info())


# #### float型を縦に結合するとデータ型が書き変わる
# > float32に変換してメモリを節約する方法もあるが、扱える桁数が少なくなり計算の精度が落ちる

# In[ ]:


# float型を含むDataFrameを作成
df2 = pd.DataFrame(index=[0, 1], columns=['value'], data=[0.0, np.nan])

# df1にdf2を追加
df3 = df1.append(df2)

# データを確認。reset_indexをするとindexが連番になりメモリが節約できる
print(df3.info(), '\n')
print(df3.reset_index(drop=True).info())


# #### メモリを節約しつつ欠損値を扱う
# 

# In[ ]:


df = df3.reset_index(drop=True).copy()

# 欠損値フラグを作成
df['NaN_flag'] = False
df['NaN_flag'] = df['NaN_flag'].mask(df['value'].isna(), True)
# bool型は2値なので非常に軽い。大きなデータを扱うときは安易に数値化しない方がよい

# 欠損値を置換し、int32型に変更
df.loc[pd.isnull(df['value']), 'value'] = -9999
df['value'] = df['value'].astype('int32')

# 欠損値をフラグ化して数値で補完することで7.6MBから4.8MBに節約できた
display(df.tail())
df.info()


# ### メモリの解放

# DataFrameはメモリ効率がよくないので、不要となったら削除してメモリ解放するとよい  
# 環境により実行結果が変わるため、ここではコードを実行しない
# ```
# del df
# gc.collect()
# ```

# ---
# ## カラムの抽出
# > 可読性とメンテナンス性を高めるテクニック

# ### カラム名に特定文字列を含まないものを抽出する（シンプルな実装）

# In[ ]:


category[[c for c in category.columns if not 'small' in c]].head()


# ### カラム名に特定文字列を含まないものを正規表現で抽出  
# > コードは増えるが応用範囲は広い

# In[ ]:


category[[c for c in category.columns if not re.match('.*small*.', c)]].head()


# ---
# ## データ型のチェック
# > 集計処理やデバッグ時に活用できる

# ### あるカラムにint以外が入っている行を表示する

# In[ ]:


product[product['unit_price'].map(lambda x: not x.is_integer())]


# ### dtypeがobjectの列名のみ抽出する

# In[ ]:


product.select_dtypes(include=object).columns.tolist()


# ### dtypeがobjectでない列名のみ抽出する

# In[ ]:


product.select_dtypes(exclude=object).columns.tolist()


# ---
# ## データの結合
# > ただのmergeやconcatではない結合方法をまとめる

# ### indicatorを使って片方にしかないレコードを明示する
# > left_onlyとなるものは元データにしか存在せず、right_onlyとなるものは結合するデータにしか存在しない

# In[ ]:


pd.merge(receipt, customer, on='customer_id', how='outer', indicator=True)    .query('_merge != "both"')[['sales_ymd', 'customer_id', '_merge']]


# ### 文字列の部分一致を条件にデータを結合する

# In[ ]:


# 顧客データに近隣店舗のデータをマージする

# store_nameをリスト化
store_names = store['store_name'].to_list()
# 顧客の住所にstore_nameが含まれるかチェック
customer['store_name'] = customer['address'].map(lambda x: [elem for elem in store_names if elem.strip('店') in x])
# store_nameが含まれていたら0番目の要素を取得し、そうでなければNoneとする
customer['store_name'] = customer['store_name'].map(lambda x: x[0] if len(x) else None)
# store_nameをキーにして、顧客データに近隣店舗のデータをマージ
customer.merge(store[['store_name', 'longitude', 'latitude']], on='store_name', how='left').head()


# ### すごく早い全結合

# In[ ]:


# 時系列データに欠損がある場合に有用
ymd = list(range(20230101, 20230104, 1))
stores = ['small', 'large']
df1 = pd.DataFrame({'ymd': np.repeat(ymd, len(stores)),
                   'stores': stores*len(ymd)})
df1


# ### すごく早い全結合（3列以上）

# In[ ]:


# 1列増やしたい場合は、ユニーク数の少ない列同士で全結合してからマージすると良い
stores = ['small', 'large']
gender = ['male', 'female']
df2 = pd.DataFrame({'stores': np.repeat(stores, len(gender)),
                   'gender': gender*len(stores)})
df1.merge(df2).sort_values(['ymd', 'stores', 'gender'])


# ### 特定のデータにフラグを立てる

# #### maskメソッドを使う方法
# > 直感的に書けるが少し冗長な印象

# In[ ]:


black_list = ['大野 あや子', '堀井 かおり']
df = customer.copy()

df['in_black_list'] = False
df['in_black_list'].mask(df['customer_name'].isin(black_list), True, inplace=True)
df.head()


# #### 新たにDataFrameを作って結合させる方法
# > black_listのテーブルを作って結合させるイメージ。コードではなくテーブルを管理すればよくなるのでメンテナンス性に優れる

# In[ ]:


bl_df = pd.DataFrame(black_list, columns=['customer_name'])
bl_df['in_black_list'] = True
customer.merge(bl_df, how='left').head()


# ---
# ## replace関数
# > 簡単な処理だったらmaskよりも短く書ける。mapとlambdaを組み合わせると応用範囲が広い  
# > 元データを変更する場合はmask同様、`inplace=True`が必要  
# > 以下はmaskとの簡単な比較
# > - `df['col'].mask(df['col'] == 0, np.nan)`  
# > - `df['col'].replace(0, np.nan)`

# ### 数字を置き換える

# In[ ]:


# gender_cdが9の行を-9999に置き換える
customer['gender_cd'].replace('9', '-9999')[:5]


# ### 文字列を置き換える

# In[ ]:


# 電話番号のハイフンを削除する
store['tel_no'].apply(lambda x: x.replace('-',''))


# ---
# ## groupbyメソッド
# > もう少しコード追加したい

# ### 集計した1つのカラムのカラム名を変更する
# > カラムが2つ以上になると対応できないので次項のコードを使う

# In[ ]:


product.groupby('category_major_cd')[(c:='unit_cost')].sum().reset_index(name=c+'_sum').head()


# ### 集計した2つ以上のカラムのカラム名を一括で変更する

# In[ ]:


df = geocode.groupby(group:=['prefecture', 'city'])[(col:=['longitude', 'latitude'])].mean().reset_index() # :=はセイウチ演算子
# カラム名を丸ごと書き換える
df.columns = group + [f'{c}_mean' for c in col]
df.head()


# ### mergeを使わずにgroupbyの集計値の列を追加する

# In[ ]:


# 追加したい集計値
store.groupby('prefecture')['floor_area'].mean().reset_index()


# In[ ]:


# transformを使うと集計値の列を追加できる
store['avg_floor_area_by_prefecture'] = store.groupby('prefecture')['floor_area'].transform('mean')
store[['store_name', 'prefecture', 'avg_floor_area_by_prefecture']].head()


# ---
# ## 縦持ち・横持ち変換

# ### データをクロス集計してインデックスを整える

# In[ ]:


# 集計用に年代を作成
customer['age_era'] = customer['age']//10*10
# クロス集計
customer_pivot = pd.pivot_table(data=customer, 
                                index=['application_store_cd', 'age_era'], 
                                columns='gender', 
                                values='customer_id', 
                                aggfunc='count').fillna(0)
# データ型を一括変換
customer_pivot = customer_pivot.astype({'不明': 'int',
                                        '女性': 'int',
                                        '男性': 'int'})
# カラムからnameを削除
customer_pivot.columns.name = None
# インデックスをリセット
customer_pivot = customer_pivot.reset_index()
customer_pivot.head()


# ---
# ## queryメソッド
# > mapやlambdaを使わずに様々な抽出ができ、True, Falseで抽出するよりも可読性が高い  
# > engine='python'のオプションを指定しないと正規表現等が動かない場合があるので注意  

# ### 値が平均値以上のものを抽出する

# In[ ]:


customer.query(f'age > age.mean()', engine='python').head()


# ### 条件式自体を変数として複雑な抽出をする
# > 条件式自体を変数に格納する場合は、条件式内で@varのような書き方はできないので、{var}と書く

# In[ ]:


# 条件式を1つ1つ定義
state_gender = 'gender == "男性" | gender == "不明"'
state_age = 'age < 30'
state_application_date = '20160000 <= application_date < 20170000'
# 条件式をリストに格納
state_list = [state_gender, state_age, state_application_date]
# 条件式を結合
print(state:=(' & ').join([f'({q})' for q in state_list]))
customer.query(state, engine='python').head()


# ### リストで抽出する

# In[ ]:


# 東京都と神奈川県の店舗を抽出する
prefecture_list = ['東京都', '神奈川県']
store.query(f'prefecture in {prefecture_list}', engine='python').head()


# In[ ]:


# 東京都と神奈川県でない店舗を抽出する
store.query(f'not prefecture in {prefecture_list}', engine='python').head()


# ### 正規表現で抽出する

# In[ ]:


# 名前に美または優を含む抽出する。repr()を使って美|優をダブルクォーテーション付きのまま代入するのがポイント
print(names:=repr('|'.join(['美', '優']) ) ) # namesの中身は'美|優'
customer.query(f'customer_name.str.contains({names})', engine='python').head()


# ### 特定カラムに欠損値がある行を抽出する

# In[ ]:


# townが欠損している行を抽出する
geocode.query('town.isna()', engine='python').head()


# ---
# ## 繰り返し処理
# > xxx

# ### 複数のDataFrameから同じ条件でレコードを削除する

# In[ ]:


# 先頭から5行ずつ抽出してDataFrameを作成
idxs = list(range(0, 10, 5))
display(df1:=customer.iloc[idxs[0]:idxs[0]+5])
display(df2:=customer.iloc[idxs[1]:idxs[1]+5])

# 女性を抽出するクエリ文を作成
q = 'gender_cd=="1"'
# 複数のdfから女性のレコードを削除
for df in [df1, df2]:
    df.drop(df.query(q).index, inplace=True)

# DataFrameのviewを変更しているため警告が出ているが、この場合は望ましい結果となっている
display(df1)
display(df2)


# ---
# ## 任意の関数でカーブフィッティングさせる
# > ホワイトボックスモデルを作成するときや時系列でナイーブ予測特徴量を作成するときに有用

# ### DataFrameでデータを定義

# In[ ]:


# リストからDataFrameを作成
x = range(0,10)
y = [1, 0.65, 0.5, 0.3, 0.3, 0.22, 0.2, 0.2, 0.18, 0.22]
dataset = pd.DataFrame(list(zip(x, y)), columns=['x', 'y'])

# 偶数行のデータを学習データとする
train_idx = [c for c in dataset.index if c%2==0]
train = dataset.iloc[train_idx]

# 評価データと学習データを定義
x_all = dataset['x'].to_numpy()
y_all = dataset['y'].to_numpy()
x_train = train['x'].to_numpy()
y_train = train['y'].to_numpy()

# 可視化
def plt_dataset(x_all, y_all, x_train, y_train):
    plt.plot(x_all, y_all, 'bo')
    plt.plot(x_train, y_train, 'ro', label='train data')
plt_dataset(x_all, y_all, x_train, y_train)
plt.legend()
plt.show()


# ### モデル定義

# In[ ]:


class curve_fit_model:
    def __init__(self):
        self.func = 'func'
        self.params = []
        self.popt = []
    def set_func(self, func, num_params):
        '''y=f(x)の形式の関数とパラメータ数を受け取って関数を定義する'''
        self.func = func
        self.params = [1]*num_params # 初期値が全て1のパラメータリストを作成
    def fit(self, x, y):
        '''xとyを受け取ってフィッティングさせる。係数が出力される'''
        popt, pcov = curve_fit(self.func, x, y, p0=self.params) # フィッティング
        coefs =[print(f'coef{i+1}: {j}') for i, j in enumerate(popt)] # 係数を出力
        self.popt = popt # 係数が入ったリスト
    def pred(self, x):
        '''xを受け取ってyを計算する'''
        y = self.func(x, *self.popt)
        return y


# ### フィッティング関数の定義

# In[ ]:


def xn(x, *params): 
    '''多項式近似'''
    y = sum(param*x**n for (n, param) in enumerate(params))
    return y

def exp(x, *params):
    '''指数関数的減衰'''
    y = params[0]*np.exp(-x/params[1])
    return y


# ### 学習結果を可視化

# In[ ]:


func_list = [
    {'func': xn, 'num_params': 4},
    {'func': exp, 'num_params': 2},
]

for func in func_list:
    # モデル学習
    print(func['func'].__doc__)
    model = curve_fit_model()
    model.set_func(**func)
    model.fit(x_train, y_train)
    y_pred = model.pred(x_all)
    # MSEを計算
    print(f'MSE: {mean_squared_error(y_all, y_pred):.2%}')
    # 予測と可視化
    plt_dataset(x_all, y_all, x_train, y_train)
    plt.plot(x_all, y_pred, 'g', label='curve fit')
    plt.legend()
    plt.show()


# ---
# ## 便利な自作関数
# > スクリプトの最初の方に定義しておくと便利

# ### DataFrameの行ごとに関数を適用する
# > 関数を2段階に分けて書くことで処理がわかりやすくなり、コードが再利用しやすくなる

# In[ ]:


def profit_ratio(price, cost):
    '''2つの値を受け取って利益率を計算する
    '''
    return (price - cost) / price

def func_to_cols(row, func, columns):
    '''dfの行の特定カラムの値に対して関数を適用する
    戻り値を*valuesとすることで、リストの中身が展開（アンパック）されて関数に渡される
    valuesの長さが合っていればfuncは3個以上の引数を引数を取れる
    '''
    values = [row[c] for c in columns]
    return func(*values)

# applyはデフォルトだと1つの引数しか受け取らないのでargsで関数名とカラム名を与える
columns = ['unit_price', 'unit_cost']
product['profit_ratio'] = product.apply(func_to_cols, args=[profit_ratio, columns], axis=1)
product.head()


# ### 形式の同じCSVデータを読み込んで縦方向に結合する関数

# In[ ]:


# for文の中でpd.concat()するよりもまとめてconcatした方が速い
def get_concat_df(folder_path, sep='*.csv'):
    df_list = []
    file_list = glob.glob(os.path.join(folder_path, sep))
    for file_path in file_list:
        df_list.append(pd.read_csv(file_path))
    df = pd.concat(df_list)
    return df


# ---
# ## その他の小ネタ

# ### 関数やメソッドに何度も書くキーワード引数を辞書型で渡す
# > to_csv()でファイルをいくつも出力するときに便利

# In[ ]:


kwds = {'index': False, 'sep': ','}
category.head().to_clipboard(**kwds) # index=False, sep=','と書いたのと同様になる


# ### DataFrameをクリップボードでやりとりする
# > indexが不要な場合はindex=Falseとする

# In[ ]:


# DataFrameの内容をクリップボードにコピーして、Excelなどに貼り付けられるようにする。
category.head().to_clipboard()
# クリップボードの中身を確認
print(pyperclip.paste())
# クリップボードからdfを作ることもできる
display(pd.read_clipboard())

# 引数sepを使うと、csv形式にもできる
category.head().to_clipboard(sep=',')
print(pyperclip.paste())


# ### DataFrameの特定行の列名と値をいい感じに表示する
# > カラム名に日本語が入っていると崩れるので工夫が必要

# In[ ]:


# 表示する行
row = 0 
# ここにカラム名と値を入れていく
text = [] 
# 文字を整形するため、文字数の最大値を取得
max_len = max([len(x) for x in category.columns.to_list()])
# 表示するテキスト作成
for k, v in category.iloc[row].to_dict().items():
    k, v = str(k), str(v)
    text += [f'{k.ljust(max_len+1)}: {v}']
text = ('\n').join(text)
print(text)


# ### カラムごとの集計値をdfの形にする
# > desicribe()で集計されないものもdfの形式で表示できる

# In[ ]:


category.nunique().reset_index(name='nunique')

