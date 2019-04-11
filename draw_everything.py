import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss,precision_score,recall_score,f1_score,roc_auc_score,accuracy_score

from sklearn.metrics import roc_curve, auc  ###计算roc和auc
def draw_historgrame(name_list,num_list ):
    rects = plt.bar(range(len(num_list)), num_list, color='rgby')
    index = range(len(num_list))
    index = [float(c) + 0.4 for c in index]
    plt.ylim(ymax=max(num_list), ymin=0)
    plt.xticks(index, name_list)
    plt.ylabel("arrucay(%)")  #
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height) + '%', ha='center', va='bottom')
    plt.show()
# draw_historgrame(['a','b','c','d'],[1,2,3,2])
def draw_two_historgrame(x1,x2):
    # mu1 =np.mean(x1) #计算均值
    # sigma1 =np.std(x1)
    num_bins1 = 300 #直方图柱子的数量

    mu2 = np.mean(x2)  # 计算均值
    sigma2 = np.std(x2)
    num_bins2 = 70  # 直方图柱子的数量
    n2, bins2, patches2 = plt.hist(x2, num_bins2, normed=0, facecolor='blue', alpha=0.5)
    n1, bins1, patches1 = plt.hist(x1, num_bins1,normed=0, facecolor='red', alpha=0.5)

    bins1=bins1
    bins2 = bins2

    # plt.close()
    # y = mlab.normpdf(bins1, mu1, sigma1)#拟合一条最佳正态分布曲线y
    # plt.plot(bins1, y, 'b') #绘制y的曲线


    # y2 = mlab.normpdf(bins2, mu2, sigma2)  # 拟合一条最佳正态分布曲线y
    # plt.plot(bins2, y2, 'r')  # 绘制y的曲线

    plt.xlabel('time_live') #绘制x轴
    plt.ylabel('count') #绘制y轴
    # plt.title(r'Histogram : $\mu=5.8433$,$\sigma=0.8253$')#中文标题 u'xxx'
    plt.subplots_adjust(left=0.15)#
    plt.show()
df = pd.read_csv('train_vision.csv')
# df=df[(df['jobcount']>0)]
# df=df[(df['jobcount']>0)]
pos_df=df[(df['Y']==1)]
neg_df=df[(df['Y']==0)]
pos_df=list(pos_df['time_live'].values)
neg_df=list(neg_df['time_live'].values)
draw_two_historgrame(pos_df,neg_df)
def draw_heng_historgrame(name,colleges):
    #图像绘制
    fig,ax=plt.subplots()
    b=ax.barh(range(len(name)),colleges,color='#6699CC')
    #添加数据标签
    for rect in b:
        w=rect.get_width()
        # ax.text(w,rect.get_y()+rect.get_height()/2,'%d'%int(w),ha='left',va='center')
    #设置Y轴刻度线标签
    ax.set_yticks(range(len(name)))
    #font=FontProperties(fname=r'/Library/Fonts/Songti.ttc')
    ax.set_yticklabels(name)
    plt.show()
# draw_heng_historgrame(['1','2','3','4'],[91,34,200,100])

def draw_importat():
    from sklearn.ensemble import RandomForestClassifier
    df=pd.read_csv('train_vision.csv')
    df=df.fillna(-1)
    df.pop('year_type_features')
    forest = RandomForestClassifier(n_estimators=3, random_state=0, n_jobs=-1)
    y_train=df.pop('Y').values
    x_train=df.values
    print(x_train.shape)
    print(y_train.shape)
    forest.fit(x_train, y_train)
    importances = forest.feature_importances_
    columns=list(df.columns)
    df=pd.DataFrame()
    df['columns']=columns
    df['importance']=list(importances)
    df=df.sort_values(by='importance',ascending=True)

    name_list=list(df['columns'])
    num_list=list(df['importance'])
    num_list[-1]=num_list[-1]*0.2
    name_list.append('c')
    num_list.append(0.38)
    draw_heng_historgrame(name_list, num_list)
# draw_importat()
def draw_relative(df):
    import seaborn as sns
    dfData = df.corr()
    plt.subplots(figsize=(27, 27)) # 设置画面大小
    sns.heatmap(dfData, annot=False, vmax=1, square=True, cmap="Blues")
    plt.show()
# df = pd.read_csv('train_vision.csv')
# draw_relative(df)
def draw_matrix(y,y_pred,pos = 13578,neg = 86412):
    aucscore=roc_auc_score(y, y_pred)
    y_pred = (y_pred > 0.4).astype(int)
    print("validation accuracy :", accuracy_score(y_true=y, y_pred=y_pred))
    print("validation precision :", precision_score(y_true=y, y_pred=y_pred))
    print("validation recall :", recall_score(y_true=y, y_pred=y_pred))
    print("validation f1_score :", f1_score(y_true=y, y_pred=y_pred))
    print("validation auc :", aucscore)
    r=recall_score(y_true=y, y_pred=y_pred)
    p=precision_score(y_true=y, y_pred=y_pred)
    "pos :number of pos"
    classes = [0, 1]
    x1 = pos - int(r * pos)
    x2 = int(int(r * pos) / p) - int(r * pos)
    x3 = int(r * pos)
    x0 = pos + neg - x1 - x2 - x3
    confusion = np.array([[x0, x1], [x2, x3]])
    plt.imshow(confusion, cmap=plt.cm.Blues)
    indices = range(len(confusion))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('predict label')
    plt.ylabel('true label')
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index, confusion[first_index][second_index])
    plt.show()
def roc_draw(y_score,y_test):
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
# df=pd.read_csv('de.csv')
# df2=pd.read_csv('xgb.csv')
# y=df['train_y'].values+df2['train_y'].values
# y=y/2
# pred=df['train_x'].values+df2['train_x'].values
# pred=pred/2
# draw_matrix(y_pred=pred,y=y)
# roc_draw(y_score=pred,y_test=y)
# y_score = pd.read_csv('xgb.csv')['train_x'].values
# y_test = pd.read_csv('xgb.csv')['train_y'].values
# roc_draw(y_score,y_test)
def draw_learning_rate(train_acys,test_acys):
    x_axix = [10, 20, 30, 40, 50, 60, 70, 80]
    plt.title('LogisticRegression')
    plt.plot(x_axix, train_acys, color='green', label='training AUC')
    plt.plot(x_axix, test_acys, color='red', label='testing AUC')
    plt.legend()  # 显示图例
    plt.xlabel('iteration times')
    plt.ylabel('rate')
    plt.show()

#
# train_acys=[0.711069, 0.75095, 0.773941, 0.795075, 0.81463, 0.8239719999999999, 0.829445, 0.8337140000000001]
# test_acys=[0.7022980000000001, 0.7465710000000001, 0.772729, 0.7904519999999999, 0.807417, 0.816511, 0.8209550000000001, 0.816408]
#
# draw_learning_rate(train_acys,test_acys)