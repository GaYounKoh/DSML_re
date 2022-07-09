# PBM : pneum_bio_marker

import pandas as pd
import numpy as np

# 결과 확인을 용이하게 하기 위한 코드
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

#경고 에러 무시
import warnings
warnings.filterwarnings('ignore')


#시각화 라이브러리
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
#한글설정
import matplotlib.font_manager as fm

font_dirs = ['/usr/share/fonts/truetype/nanum', ]
font_files = fm.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    fm.fontManager.addfont(font_file)
    
# 한글 출력을 위해서 폰트 옵션을 설정합니다.
# "axes.unicode_minus" : 마이너스가 깨질 것을 방지

sns.set(font="NanumBarunGothic", 
        rc={"axes.unicode_minus":False},
        style='darkgrid')
plt.rcParams["legend.framealpha"]= 0.5 # legend 투명도


class PBM:
    
    def item_search_prepare(self):
        mimic_path = '/data/MIMIC_III/'
        # LABEVENTS
        # ITEMID로 LABEL, LOINC_CODE 찾기
        lab = pd.read_csv(mimic_path + 'D_LABITEMS.csv')
        lab = lab[['ITEMID', 'LABEL', 'LOINC_CODE']]
        lab['TYPE'] = 'LAB'


        # PRESCRIPTIONS
        # NDC로 DRUG, DRUG_NAME_POE 찾기
        pre = pd.read_csv(mimic_path + 'PRESCRIPTIONS.csv')
        pre = pre[['NDC', 'DRUG']]
        # 'FORMULARY_DRUG_CD'는 안쓰기로 함.
        # 'DRUG_NAME_POE'도 굳이...
        pre['LOINC_CODE']=0
        pre.rename(columns = {'NDC':'ITEMID', 'DRUG':'LABEL'}, inplace = True)
        pre['TYPE'] = 'PRE'


        # PROCEDURES
        # ITEMID로 LABEL 찾기
        pro = pd.read_csv(mimic_path + 'D_ITEMS.csv')
        pro = pro[['ITEMID', 'LABEL']]
        pro['LOINC_CODE'] = 0
        pro['TYPE'] = 'PRO'


        # lab, pre, pro 합치기
        items = pd.concat([lab,pre,pro])
        items = items[items['ITEMID'].notnull()]
        items['ITEMID'] = items['ITEMID'].astype(int)
        return items
        
        
        
    def absum_prepare(self):
        ### absum prepare
        # 0. 4069개 종류의 feature name 정보를 담은 dict 생성
        path = '/project/LSH/** 해외_Journal of Biomedical Informatics/'
        feature_df = pd.read_csv(path+'feature_df.csv')
        feature_name = dict(zip(feature_df['feature'], feature_df['feature_name']))

        X = np.load(path+'x_(7727,10,3595).npy')
        y = np.load('/project/LSH/y_(7727,1).npy')
        COLS = list(pd.read_csv(path+'total_data_7727_10_3595.csv')['ITEMID'].sort_values().unique())

        # 1. 사망 / 생존 환자 인덱스
        d_index = np.where(y==1)[0] # 사망
        s_index = np.where(y==0)[0] # 생존

        # 2. 사망 / 생존 환자 분리
        d_X = X[d_index]
        s_X = X[s_index]

        result = []
        result_s = []

        for d in range(10):
            for f in range(d_X.shape[-1]):
                d_sum = d_X[:,d,f].sum()/d_X.shape[0]
                s_sum = s_X[:,d,f].sum()/s_X.shape[0]
                result.append({'cols':COLS[f], 'day':10-d,'per':d_sum})
                result_s.append({'cols':COLS[f], 'day':10-d,'per':s_sum})

        d_df = pd.DataFrame(result).sort_values(['cols','day']).reset_index(drop=True)
        s_df = pd.DataFrame(result_s).sort_values(['cols','day']).reset_index(drop=True)
#         d_df.shape, s_df.shape
        
        return feature_name, d_df, s_df
    
    
    
    def item_search(self, items, itemid):
        type_ = items[items['ITEMID']==itemid]['TYPE'].unique()
        label_ = items[items['ITEMID']==itemid]['LABEL'].unique()
        return type_, label_
    
    
    def absum(self, top10_list, feature_name, d_df, s_df):
        plt.figure(figsize = (12,10), dpi=150)
        i = 0
        for f in top10_list:
            if f == 0:
                continue
            plt.subplot(4,4,1+i)
            plt.title(feature_name[f])
            ax = sns.lineplot(data = d_df[d_df['cols']==int(f)], x = 'day', y='per', label='사망')
            ax = sns.lineplot(data = s_df[s_df['cols']==int(f)], x = 'day', y='per', label='생존', linestyle=':', marker='o')
            ax.invert_xaxis()
            ax.legend(loc='upper left')
            i += 1
        plt.tight_layout()
        
        
        