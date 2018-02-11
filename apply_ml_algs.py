import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                    ExtraTreesClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from xgboost.sklearn import XGBClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing import sequence
from keras.optimizers import Adam, Adadelta

dt_params = {'max_features': 18,
            'splitter': 'best',
            'criterion': 'gini',
            'max_depth': 4,
            'min_samples_leaf': 8}

xt_params = {'min_samples_leaf': 1,
            'min_samples_split': 5,
            'n_estimators': 150}

gbt_params = {'min_samples_split': 2,
            'loss': 'exponential',
            'learning_rate': 0.5,
            'n_estimators': 100,
            'max_depth': 2}

rf_params = {'min_samples_split': 2,
            'n_estimators': 100,
            'criterion': 'gini',
            'max_depth': None,
            'min_samples_leaf': 2}

svm_params = {'C': 22, 'gamma': 0.1}

xgb_params = {'n_estimators': 200,
            'objective': 'binary:logistic',
            'learning_rate': 0.2,
            'max_depth': 3,
            'gamma': 0}

dt_model = DecisionTreeClassifier(**dt_params)
xt_model = ExtraTreesClassifier(**xt_params)
gbt_model = GradientBoostingClassifier(**gbt_params)
knn_model = KNeighborsClassifier()
logreg_model = LogisticRegression(C=10000)
rf_model = RandomForestClassifier(**rf_params)
svm_model = SVC(**svm_params)
xgb_model = XGBClassifier(**xgb_params)

np.random.seed(7)

MAX_LEN = 50

pos = pd.read_pickle("pos.pkl")
neg = pd.read_pickle("neg.pkl")

unique_diags = pos.diag_cd.unique()
diag2id = {i:j for i,j in zip(unique_diags, np.arange(len(unique_diags)))}

pos_members = pos.sys_id.unique().tolist()
neg_members = neg.sys_id.unique().tolist()

pos_data = dict()
neg_data = dict()
for p in pos_members:
    diags = pos[pos.sys_id == p].diag_cd.tolist()
    diags = [diag2id[d] for d in diags]
    pos_data[p] = diags

for n in neg_members:
    diags = neg[neg.sys_id == n].diag_cd.tolist()
    diags = [diag2id[d] for d in diags]
    neg_data[n] = diags

pos_data_array = sequence.pad_sequences(pos_data.values(), maxlen=MAX_LEN)
neg_data_array = sequence.pad_sequences(neg_data.values(), maxlen=MAX_LEN)

rnn_data = np.vstack((pos_data_array, neg_data_array))
rnn_label = np.array([1]*pos_data_array.shape[0] + [0]*neg_data_array.shape[0])

df = pd.read_pickle("data.pkl")

data = df.drop('Target', axis=1)
target = df.Target

dt_acc = []
xt_acc = []
gbt_acc = []
knn_acc = []
logreg_acc = []
lstm_acc = []
rf_acc = []
svm_acc = []
xgb_acc = []

base_rf_acc = []

N = 10
for i in xrange(N):
    Xtrain, Xtest, ytrain, ytest = train_test_split(data, target, train_size=.7)

    base_rf_acc.append(RandomForestClassifier().fit(Xtrain, ytrain).score(Xtest, ytest))
    dt_acc.append(dt_model.fit(Xtrain, ytrain).score(Xtest, ytest))
    xt_acc.append(xt_model.fit(Xtrain, ytrain).score(Xtest, ytest))
    gbt_acc.append(gbt_model.fit(Xtrain, ytrain).score(Xtest, ytest))
    knn_acc.append(knn_model.fit(Xtrain, ytrain).score(Xtest, ytest))
    logreg_acc.append(logreg_model.fit(Xtrain, ytrain).score(Xtest, ytest))
    rf_acc.append(rf_model.fit(Xtrain, ytrain).score(Xtest, ytest))
    svm_acc.append(svm_model.fit(Xtrain, ytrain).score(Xtest, ytest))
    xgb_acc.append(xgb_model.fit(Xtrain, ytrain).score(Xtest, ytest))

for i in xrange(N):
    rnn_Xtrain, rnn_Xtest, rnn_ytrain, rnn_ytest = train_test_split(rnn_data, rnn_label, train_size=.7)

    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(20, 20, input_length=MAX_LEN))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    adam = Adam(lr=0.01, decay=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    history = model.fit(rnn_Xtrain, rnn_ytrain, nb_epoch=100, batch_size=100, verbose=0)
    scores = model.evaluate(rnn_Xtest, rnn_ytest, verbose=0)
    lstm_acc.append(scores[1])

    #lstm_acc.append(np.random.uniform(low=.8, high=.9))

mask = np.argsort([np.mean(dt_acc), np.mean(xt_acc), np.mean(gbt_acc), np.mean(knn_acc), np.mean(logreg_acc),
                np.mean(lstm_acc), np.mean(rf_acc), np.mean(svm_acc), np.mean(xgb_acc)])[::-1]

a = np.vstack((dt_acc, xt_acc, gbt_acc, knn_acc, logreg_acc, lstm_acc, rf_acc, svm_acc, xgb_acc))
col = np.array(["DT", "XT", "GBT", "KNN", "LOGREG", "LSTM", "RF", "SVM_RBF", "XGB"])

a = a[mask]
col = col[mask]

accuracies = pd.DataFrame(a.T, columns=col)

rcParams['figure.figsize']= (10,6)
accuracies.plot(kind='box')
plt.title("Accuracies ({0} Trials)".format(N))
plt.show()
#plt.savefig("acc.png")
