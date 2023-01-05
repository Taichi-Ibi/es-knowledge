import glob
import json
import re
import shutil
import subprocess

def main():
    files = glob.glob(nb_path:='script/ipynb/*.ipynb')
    for file in files:
        del_empty_cells(file)
        init_exec_cnt(file)
    export_py(nb_path)
    files = glob.glob(py_path:='script/ipynb/*.py')
    for file in files:
        new_file = re.sub('/ipynb/', '/py/', file)
        move_py(file, new_file)
    return

def del_empty_cells(file):
    with open(file) as f: # json読み込み
        jsn = json.load(f)
    cells = [] # 空のリストを生成
    for c in jsn['cells']: # セルごとにforループ
        if (c['source'] != []): # セルが空じゃない場合
            cells.append(c) # 新しいリストに追加する
    jsn['cells'] = cells # 中身を入れ替える
    with open(file, 'w') as f: # json書き込み
        json.dump(jsn, f)
    return

def init_exec_cnt(file):
    with open(file) as f: # ファイルをtxtとして読み込み
        txt = f.read()
    txt = re.sub('"execution_count": [0-9]{1,}', '"execution_count": null', txt) # execution_countが数値だったらnullにする
    txt = re.sub('"executionCount": [0-9]{1,}', '"executionCount": null', txt)
    with open(file, 'w') as f: # ファイル書き込み
        f.write(txt)
    return

def export_py(nb_path):
    subprocess.run(['jupyter', 'nbconvert', '--to', 'python', nb_path])
    return

def create_path(directory):
    py = f'script/{directory}/*.py'
    return py

def move_py(file, new_file):
    shutil.move(file, new_file)
    return

if __name__ == "__main__":
    main()