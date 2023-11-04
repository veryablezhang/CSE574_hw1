import numpy as np
import pandas as pd
import matplotlib.pyplot as py
import seaborn as sns
import math

def wine():
    wine = pd.read_csv('datasets/winequality-red.csv')
    print(wine.head())
    print(wine.shape)
    print(wine.describe(include = 'all'))
    missing_vals = wine.isnull().sum()
    print(missing_vals)

    print(wine.corr())

    sns.set_theme()
    sns.jointplot(x='volatile acidity',y='quality', data=wine)
    py.show()
    sns.jointplot(x='alcohol',y='quality', data=wine)
    py.show()

    shuffled = wine.sample(frac=1)
    total_rows = wine.shape[0]
    train_size = int(total_rows*0.8)

    train = shuffled[0:train_size]
    test = shuffled[train_size:]
    target= 'pH'
    X_train = train.drop(['quality',target],axis=1)
    y_train = train[target]
    X_test = test.drop(['quality',target],axis=1)
    y_test = test[target]

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    X = X_train.values
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_train.values)
    pred = w.T.dot(X_test.values.T)
    print('w: ',w)
    MSE = 1/y_test.shape[0]*(y_test.values-X_test.values.dot(w)).T.dot(y_test.values-X_test.values.dot(w))
    print('MSE: ',MSE)

    py.scatter(y_test,pred)
    py.xlabel('Actual pH')
    py.ylabel('Predicted pH')
    py.title('Prediction vs Ground truth')
    py.show()


def penguin():
    Penguin = pd.read_csv('datasets/penguins.csv')
    print(Penguin.head())
    print(Penguin.shape)
    print(Penguin.describe(include = 'all'))
    missing_vals = Penguin.isnull().sum()
    print(missing_vals)

    num_missing_rows = Penguin.isnull().any(axis =1).sum()
    print(num_missing_rows)
    Penguin.dropna(axis =0, inplace = True)
    Penguin.drop(['year'],axis = 1, inplace = True)
    num_cols = Penguin.columns[(Penguin.dtypes != 'object').tolist()].tolist()
    for column in Penguin[num_cols]:
        min = Penguin[column].min()
        max = Penguin[column].max()
        Penguin[column] = (Penguin[column]-min)/(max-min)

    Penguin = pd.concat([Penguin, pd.get_dummies(Penguin[['species','island']])], axis = 1)
    Penguin.drop(['species','island'], axis = 1, inplace = True)

    print(Penguin['sex'].value_counts())
    Penguin = Penguin.replace(to_replace=['male','female'], value=[0,1])

    print(Penguin.corr())

    # sns.jointplot(x='bill_length_mm',y='sex', data=Penguin)
    # py.show()
    # sns.jointplot(x='bill_depth_mm',y='sex', data=Penguin)
    # py.show()
    # sns.jointplot(x='flipper_length_mm',y='sex', data=Penguin)
    # py.show()
    # sns.jointplot(x='body_mass_g',y='sex', data=Penguin)
    # py.show()

    shuffled = Penguin.sample(frac=1)
    total_rows = Penguin.shape[0]
    train_size = int(total_rows*0.8)

    train = shuffled[0:train_size]
    test = shuffled[train_size:]
    X_train = train.drop(['sex'],axis=1)
    y_train = train['sex']
    X_test = test.drop(['sex'],axis=1)
    y_test = test['sex']
    
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    model = LogitRegression()
    model.fit(X_train, y_train)
    loss = model.loss
    py.plot(np.array(range(model.iter))+1, loss)
    py.xlabel('Epoch')
    py.ylabel('Loss')
    py.show()

    pred = model.predict(X_test)
    corrects = [1 if pred[i]==y_test.values[i] else 0 for i in range(y_test.shape[0])]
    accuracy =  sum(corrects)/y_test.shape[0]
    print(accuracy)

class LogitRegression():
    def __init__(self, lr=0.000001, iter=100000) -> None:
        self.lr = lr
        self.iter = iter
        return
    
    def fit(self,X_train,y_train):
        self.w = np.random.uniform(0,1,X_train.shape[1])
        self.b = 0
        self.loss = []
        print('start fitting on ', X_train.shape[0], ' samples.' )
        for epoch in range(self.iter):
            if (epoch+1)%1000==0:
                print('epoch: ', epoch+1, '/', self.iter)
            self.gradient_descent(X_train,y_train)
            loss = self.cost(X_train,y_train)
            self.loss.append(loss)
        return 

    def sigmoid(self,X):
        return np.array([1/(1+math.exp(-z)) for z in (X.dot(self.w)+self.b)])

    def cost(self, X, y):
        cost = -y.dot(X.dot(self.w)+self.b) + sum([np.log(1+math.exp(z)) for z in (X.dot(self.w)+self.b)])
        return cost

    def gradient_descent(self, X, y):
        p1 = self.sigmoid(X)
        gradient_w = -X.T.dot(y-p1)
        gradient_b = -sum(y-p1)
        self.w = self.w - self.lr*gradient_w
        self.b = self.b - self.lr*gradient_b

    def predict(self,X):
        return np.array([1 if p >= 0.5 else 0 for p in self.sigmoid(X)])

if __name__ == "__main__":
    # wine()
    penguin()




