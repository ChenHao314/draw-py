import pandas as pd
import os
import copy
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.linear_model import LinearRegression,LogisticRegression,LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import log_loss,precision_score,recall_score,f1_score,roc_auc_score,accuracy_score
from  collections import  Counter
import time
### data process ######################################
data_root='data/'
# data_root=''
baseinfoPath=os.path.join(data_root,'1baseinfo.csv')
alterinfoPath=os.path.join(data_root,'2alterinfo.csv')
jobinfoPath=os.path.join(data_root,'3jobinfo.csv')
trainPath=os.path.join(data_root,'train.csv')
evaluationPath=os.path.join(data_root,'evaluation.csv')
def str_process(x):
    x=str(x)+'&#'
    return x
def add_jobinfo(jobinfo,df):
    min_date = jobinfo['PublishDate'].values.min()
    EID2jobactive=Counter()
    EID2jobactive.update(list(jobinfo['EID']))
    EID2PublishDate = dict(zip(list(jobinfo['EID']), list(jobinfo['PublishDate'])))
    EID = df['EID']
    PublishDate = []
    for id in EID:
        try:
            date = EID2PublishDate[id]
        except KeyError:
            date = min_date
        PublishDate.append(date)
    jobactive = []
    for id in EID:
        try:
            active = EID2jobactive[id]
        except KeyError:
            active = 0
        jobactive.append(active)

    jobinfo=jobinfo[(jobinfo['PublishDate']>201500)]
    EID2jobactive = Counter()
    EID2jobactive.update(list(jobinfo['EID']))
    one_year_jobactive = []
    for id in EID:
        try:
            active = EID2jobactive[id]
        except KeyError:
            active = 0
        one_year_jobactive.append(active)


    jobinfo = jobinfo[(jobinfo['PublishDate'] > 201400)]
    EID2jobactive = Counter()
    EID2jobactive.update(list(jobinfo['EID']))
    two_year_jobactive = []
    for id in EID:
        try:
            active = EID2jobactive[id]
        except KeyError:
            active = 0
        two_year_jobactive.append(active)

    df['PublishDate'] = PublishDate
    df['jobcount']=jobactive
    df['oneyear_jobcount']=one_year_jobactive
    df['twoyear_jobcount']=two_year_jobactive
    return df
def alternumber_process(number):
    number=str(number)
    if number=='05':
        return 1
    else:
        return 0
def add_alterinfo(alterinfo, df):
    AlterDate = []
    min_date = alterinfo['AlterDate'].values.min()
    EID2alteractive = Counter()
    EID2alteractive.update(list(alterinfo['EID']))
    EID2AlterDate = dict(zip(list(alterinfo['EID']), list(alterinfo['AlterDate'])))
    EID = df['EID']
    for id in EID:
        try:
            date = int(EID2AlterDate[id])
        except KeyError:
            date = int(min_date)
        AlterDate.append(date)
    df['AlterDate'] = AlterDate
    alteractive = []
    for id in EID:
        try:
            active = EID2alteractive[id]
        except KeyError:
            active = 0
        alteractive.append(active)
    df['altercount'] = alteractive
    AlterNumber = ['13', '10', '12', '01', '14', '05', '99', '03', '02', '27', '04', 'A_015']
    for alter in AlterNumber:
        alter=str(alter)
        alters=[]
        for id in EID:
            try:
                alternumber = alter_counter[str(id)+'&#'+alter+'&#']
            except KeyError:
                alternumber = 0
            alters.append(alternumber)
        df[alter]=alters
    EID2capitalchange={}
    for a, b ,id in zip(list(alterinfo['AlterBefore']), list(alterinfo['AlterAfter']),list(alterinfo['EID'])):
        try:
            a=float(a)
            b=float(b)

            change = b-a
        except ValueError:
            change=0
        EID2capitalchange[id]=change
    capitalchange = []
    for id in EID:
        try:
            change = EID2capitalchange[id]
        except KeyError:
            change = 0
        capitalchange.append(change)
    df['capitalchange'] = capitalchange
    return df
def dateProcess(date):
    date=str(date).split('-')
    date=int(date[0]+date[1])
    return date
def add_baseinfo(baseinfo,df):
    Y=[]
    EID=baseinfo['EID']
    EID2Y = dict(zip(list(df['EID']), list(df['Y'])))
    for id in EID:
        try:
            y=EID2Y[id]
        except KeyError:
            y=-1
        Y.append(y)
    baseinfo['Y']=Y
    baseinfo=baseinfo[(baseinfo['Y']!=-1)].reset_index(drop=True)
    return baseinfo
def get_altercounter():
    alterInfo = pd.read_csv(alterinfoPath)
    alterInfo['strEID']=alterInfo['EID'].apply(str_process)
    alterInfo['strAlterNumber']=alterInfo['AlterNumber'].apply(str_process)
    alterInfo['alternum']=alterInfo['strEID']+alterInfo['strAlterNumber']
    alternum=list(alterInfo['alternum'].values)
    alter_counter=Counter()
    alter_counter.update(alternum)
    return alter_counter
def get_enddate():
    train=pd.read_csv(trainPath)
    train=train.fillna(-1)
    train=train[(train['EndDate'])!=-1]
    date=list(train['EndDate'])
    EID=list(train['EID'])
    return dict(zip(EID,date))
def year_feature(year):
    if year>2014:
        return 1
    elif year>2013:
        return 2
    elif year>2012:
        return 3
    elif year>2011:
        return 4
    elif year>2010:
        return 5
    elif year>2000:
        return 6
    elif year>1990:
        return 7
    elif year>1980:
        return 8
    elif year>1970:
        return 9
    elif year>1960:
        return 10
    else:
        return 11
def capital_feature(captital):
    if captital>500:
        return 51
    else:
        return int(captital/10)
def add_feature(df):
    EID=list(df['EID'])
    EID2Date=get_enddate()
    year_features=[]
    capital_features=[]
    capital_year_features=[]
    capital_type_features=[]
    capital_trade_features=[]
    year_type_features=[]
    time_live=[]
    years=list(df['CreateYear'])
    captials=list(df['RegisteredCapital'])
    types=list(df['Type'])
    type_sets=list(set(types))
    type2captial={}
    for type_ in type_sets:
        df_ =df[(df['Type']==type_)]
        type2captial[type_]=df_['RegisteredCapital'].values.mean()
    tradetypes = list(df['TradeType'])
    tradetype_sets = list(set(tradetypes))
    tradetypes2captial = {}
    for type_ in tradetype_sets:
        df_ = df[(df['TradeType'] == type_)]
        tradetypes2captial[type_] = df_['RegisteredCapital'].values.mean()
    for year_ ,captal_,type_ ,trade_ ,id_ in zip(years,captials,types,tradetypes,EID):
        year_features.append(year_feature(year_))
        capital_features.append(capital_feature(captal_))
        capital_year_features.append(year_feature(year_)*year_feature(year_))
        capital_type_features.append((captal_+15)/(type2captial[type_]+15))
        capital_trade_features.append((captal_ + 15) / (tradetypes2captial[trade_] + 15))
        year_type_features.append(str(year_feature(year_))+'&#&'+str(type_))
        try:
            time_=int(int(EID2Date[id_])/100)-year_
        except KeyError:
            time_=int(201708.0/100-year_)
        time_live.append(time_)
    df['capital_year_features']=capital_year_features
    df['capital_type_features']=capital_type_features
    df['capital_trade_features']=capital_trade_features
    # df['year_type_features']=year_type_features
    df['time_live']=time_live
    return df
def load_train_val():
    baseInfo = pd.read_csv(baseinfoPath)
    alterInfo = pd.read_csv(alterinfoPath)
    alterInfo['AlterDate'] = alterInfo['AlterDate'].apply(dateProcess)
    jobInfo = pd.read_csv(jobinfoPath)
    jobInfo['PublishDate'] = jobInfo['PublishDate'].apply(dateProcess)
    train=pd.read_csv(trainPath)
    evaluation=pd.read_csv(evaluationPath)
    train = add_baseinfo(baseInfo, train)
    evaluation = add_baseinfo(baseInfo, evaluation)
    train=add_alterinfo(alterInfo,train)
    evaluation=add_alterinfo(alterInfo,evaluation)
    train=add_jobinfo(jobInfo,train)
    evaluation=add_jobinfo(jobInfo,evaluation)
    train=add_feature(train)
    evaluation=add_feature(evaluation)
    train.pop('Feature5')
    evaluation.pop('Feature5')
    train.pop('Feature3')
    evaluation.pop('Feature3')
    train.pop('EID')
    evaluation.pop('EID')
    train.pop('capitalchange')
    evaluation.pop('capitalchange')
    # train.pop('CreateYear')
    # evaluation.pop('CreateYear')
    return train,evaluation
alter_counter=get_altercounter()
train,evaluation=load_train_val()
#






### models ######################################
class XgbModel:
    def __init__(self):
        self.params={'objective': 'binary:logistic',
                     'eta': 0.6,
                     'max_depth': 6,
                     'silent': 1,
                     'eval_metric': 'auc'}
        self.num_boost_round=12
        self.model=None
    def fit(self,train_x,train_y):
        X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.02, random_state=0)
        d_train = xgb.DMatrix(X_train, label=y_train)
        d_valid = xgb.DMatrix(X_valid, label=y_valid)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        self.model = xgb.train(self.params, d_train, self.num_boost_round, watchlist,  maximize=True, verbose_eval=2)
    def predict(self,test_x):
        d_test = xgb.DMatrix(test_x)
        pred=self.model.predict(d_test)
        return pred
class ligModel:
    def __init__(self):
        self.params =  {
            'objective': 'binary',
            'metric': 'auc',
            'boosting': 'gbdt',
            'learning_rate': 0.06,
            'verbose': 0,
            'num_leaves': 20,
            'bagging_fraction': 0.95,
            'bagging_freq': 1,
            'bagging_seed': 1,
            'feature_fraction': 0.9,
            'feature_fraction_seed': 1,
            'max_bin': 128,
            'max_depth': 8,
            'num_rounds': 90,}
        self.model=None
    def fit(self,train_x,train_y):
        X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.02, random_state=0)
        train_data = lgb.Dataset(X_train,
                                 label=y_train)
        val_data = lgb.Dataset(X_valid,
                               label=y_valid)
        self.model = lgb.train(self.params, train_data, 20, valid_sets=[val_data])
    def predict(self,test_x):
        pred=self.model.predict(test_x)
        return pred
class LR:
    def __init__(self):
        self.model = LogisticRegressionCV(multi_class="ovr", fit_intercept=True, Cs=np.logspace(-2, 2, 20), cv=4, penalty="l2",
                          solver="lbfgs", tol=0.01)
        self.ss=StandardScaler()
    def fit(self, train_x, train_y):
        train_x=self.ss.fit_transform(train_x)
        self.model.fit(train_x, train_y)
    def predict(self, test_x):
        test_x=self.ss.transform(test_x)
        pred = self.model.predict_proba(test_x)
        pred=pred[:,1]
        return pred



### stacking ######################################
model3=XgbModel()
model4=ligModel()
model0=XgbModel()
model0.num_boost_round=1
model3.num_boost_round=15
model4.params['num_leaves']=30
model1=XgbModel()
model1.num_boost_round=2
model2=ligModel()
model1.num_boost_round=3
model5=LR()
# model0=RandomForestClassifier(n_estimators=5, random_state=0, n_jobs=5)
# model1=RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=3)
# model2=RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=2)


def stacking(models,train_df,test_df,kfolds=20,fillna_value=-1,target='Y',onehot_Encoding=['Type', 'TradeType','Feature1',  'Feature2',]):
    train = train_df.fillna(fillna_value)
    evaluation = test_df.fillna(fillna_value)
    df = pd.concat([train, evaluation])
    df = pd.get_dummies(df, columns=onehot_Encoding)
    train_lenth=len(train)
    train = df[0:train_lenth]
    evaluation = df[train_lenth:]
    X = np.array(train.drop([target], axis=1))
    y = train['Y'].values
    test_X = np.array(evaluation.drop(['Y'], axis=1))
    splits = list(StratifiedKFold(n_splits=kfolds, shuffle=False, random_state=2019).split(X, y))
    train_preds=[]
    test_preds=[]
    for model_ in models:
        p_test = np.zeros((len(test_X)))
        y_pred = np.zeros((len(y)))
        for idx, (train_idx, valid_idx) in enumerate(splits):
            model=copy.deepcopy(model_)
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val =   X[valid_idx]
            y_val =   y[valid_idx]
            model.fit(X_train,y_train)
            p_test    +=    model.predict(test_X)
            pred_val_y =    model.predict(X_val)
            y_pred[valid_idx] = pred_val_y.reshape(-1)
        train_preds.append(y_pred)

        result = pd.DataFrame()
        result['train_x'] = list(y_pred)
        result['train_y'] = list(y)
        result.to_csv(str(time.time())+'.csv', index=False)
        test_preds.append(p_test/kfolds)
    scores=[]
    for pred in train_preds:
        scores.append(roc_auc_score(y, pred))
        print(" auc :", roc_auc_score(y, pred))
        print(" loss :", log_loss(y_true=y, y_pred=pred))
        pred = (pred > 0.5).astype(int)
        print(" accuracy :", accuracy_score(y_true=y, y_pred=pred))
        print(" precision :", precision_score(y_true=y, y_pred=pred))
        print("recall :", recall_score(y_true=y, y_pred=pred))
        print("f1_score :", f1_score(y_true=y, y_pred=pred))
    print("auc scores of all models:",scores)
    stack_train_X=np.array(train_preds).T
    reg = LinearRegression().fit(stack_train_X, y)
    weights=reg.coef_
    print("weights  of models:",weights)
    y_pred = np.zeros((len(y)))
    for i in range(len(weights)):
        y_pred+=stack_train_X.T[i]*weights[i]

    result = pd.DataFrame()
    result['train_x'] = list(y_pred)
    result['train_y'] = list(y)
    result.to_csv( 'cross_validation.csv', index=False)
    print("validation auc :", roc_auc_score(y, y_pred))
    print("validation loss :",log_loss(y_true=y,y_pred=y_pred))
    y_pred=(y_pred>0.5).astype(int)
    print("validation accuracy :",accuracy_score(y_true=y,y_pred=y_pred))
    print("validation precision :",precision_score(y_true=y,y_pred=y_pred))
    print("validation recall :",recall_score(y_true=y,y_pred=y_pred))
    print("validation f1_score :",f1_score(y_true=y,y_pred=y_pred))
    preds= np.zeros((len(test_X)))
    for i in range(len(weights)):
        preds+=test_preds[i]*weights[i]
    return preds

preds=stacking(models=[model0,model1,model2,model3,model4],kfolds=20,train_df=train,test_df=evaluation)






### save result ######################################
evaluation=pd.read_csv(evaluationPath)
evaluation['Y_pro']=list(preds)
if not os.path.exists('output/'):
    os.mkdir('output/')
evaluation.to_csv('output/evaluation.csv',index=False)
