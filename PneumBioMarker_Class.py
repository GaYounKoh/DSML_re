# PBM : pneum_bio_marker
class PBM:
#     itemid = 0
#     def __init__(self, itemid):
#         self.itemid = itemid
        
    
    def prepare(self):
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
        
    def item_search(self, items, itemid):
        type_ = items[items['ITEMID']==itemid]['TYPE'].unique()
        label_ = items[items['ITEMID']==itemid]['LABEL'].unique()
        return type_, label_