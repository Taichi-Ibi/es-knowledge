#!/usr/bin/env python
# coding: utf-8

# # SQLの応用的なコード

# In[ ]:


import sys; sys.path.append('../../')
from module.pandasql import check_db


# > データの読み込み。ここはいじっちゃダメ

# In[ ]:


check_db() # ローカルにdbがなければ作成する。所要時間1分弱


# In[ ]:


get_ipython().run_line_magic('load_ext', 'sql')
get_ipython().run_line_magic('sql', 'sqlite:///../../sample_data/db/sample_data.db')


# ---
# ## Title
# > explanation

# In[ ]:


get_ipython().run_cell_magic('sql', '', '\nSELECT\n    *\nFROM\n    customer\nLIMIT\n    10\n;')

