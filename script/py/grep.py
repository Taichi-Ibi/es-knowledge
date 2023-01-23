import argparse
import glob
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--word', type=str, required=True,
        help='検索する文字列')
    parser.add_argument('-p', '--ptn', type=str, required=True,
        help='検索対象とするファイル')
    parser.add_argument('-e', '--env', type=str, default='USER_ZDOTDIR', 
        help='親ディレクトリとする環境変数')
    parser.add_argument('-a', '--after', type=int, default=0,
        help='追加で下に表示する行数')
    parser.add_argument('-b', '--before', type=int, default=0, 
        help='追加で上に表示する行数')
    parser.add_argument('-r', '--result', action='store_false',
        help='オプションをつけるとマッチした行を表示しない')
    parser.add_argument('-i', '--index', action='store_false',
        help='オプションをつけるとマッチした行番号を表示しない')
    parser.add_argument('-s', '--sep', action='store_false',
        help='オプションをつけるとマッチごとに行を区切らない')
    parser.add_argument('-n', '--name', action='store_false',
        help='オプションをつけるとファイル名を出力しない')
    parser.add_argument('-m', '--mark', action='store_false',
        help='オプションをつけるとマッチした行に*を付けない')
    parser.add_argument('-c', '--count', action='store_false',
        help='オプションをつけるとマッチした行数を表示しない')
    args = parser.parse_args()
    return args

def get_matched_idxs(lines, string):
    '''サーチする文字列が含まれるリストの番号を返す'''
    idxs_matched = []
    for idx, line in enumerate(lines):
        if string in line: idxs_matched.append(idx)
        else: pass
    return idxs_matched

def collect_neighbor(num_list, n_before, n_after):
    '''リストの各数値の前後nの範囲の数値を追加して重複を除外する
    >>> collect_neighbor([2, 10], 1, 2)
    [1, 2, 3, 4, 9, 10, 11, 12]
    '''
    collected_list = []
    for num in num_list:
        for diff in range(-n_before, n_after+1):
            collected_list.append(num + diff)
    collected_list = sorted(list(set(collected_list)))
    return collected_list

def main():
    # 引数をパース
    args = get_args()

    # 環境変数を使った親ディレクトリと、パターンを結合
    pattern = os.path.join(os.getenv(args.env), args.ptn)
    # サーチするパスのリストを取得
    paths = glob.glob(pattern, recursive=True)

    # 検索結果を辞書に追加
    result = []
    for path in paths:
        with open(path, 'r') as f:
            lines = f.read().split('\n')
            r_dict = {}
            r_dict['path'] = path
            r_dict['lines'] = lines
            # マッチしたindexを取得
            r_dict['index'] = get_matched_idxs(lines, args.word)
            r_dict['count'] = len(r_dict['index'])
            # 前後の行を取得する
            r_dict['index_added'] = collect_neighbor(
                r_dict['index'], n_before=args.before, n_after=args.after)
            # ファイルの行数からはみ出たものは除外
            r_dict['index_added'] = \
                [i for i in r_dict['index_added'] if i in range(0, len(lines))]
            # 検索結果を辞書に追加
            result.append(r_dict)

    # 出力を作成
    big_output = []
    for r in result:
        if r.get('count'):
            small_output = []
            # -nの処理 ファイル名を表示
            if args.name:
                path = '- ' + r.get('path')
                # -cの処理 マッチ数を表示
                if args.count:
                    path += ' ' + str(r.get('count'))
                small_output.append(path)
            # -rの処理 検索結果を表示
            if args.result:
                idxs = r.get('index_added')
                for itr,idx in enumerate(idxs):
                    line = r.get('lines')[idx]
                    # -iの処理 行番号を表示
                    if args.index: 
                        line = '  '.join([str(idx),line])
                    # -mの処理 マッチした行をハイライト
                    if not args.mark: 
                        pass
                    elif (args.mark) & (idx in r.get('index')):
                        line = '* ' + line
                    else:
                        line = '  ' + line
                    # -sの処理 1/2 マッチごとに改行
                    if (args.sep) & (itr!=0) & ((idxs[itr]-idxs[itr-1])!=1):
                        small_output.append('\n')
                    # 行を出力リストに追加
                    small_output.append(line)
                # -sの処理 2/2 ファイルごとに改行
                if args.sep:
                    small_output.append('\n')
            # 検索結果をリストに追加
            big_output.append(small_output)

    # 出力
    for big in big_output:
        for small in big:
            print(small)

if __name__ == '__main__':
    main()