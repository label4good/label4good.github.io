# Libraires needed
# pyrebase (https://github.com/thisbejim/Pyrebase):
# pip install pyrebase
# Scikit-learn (https://scikit-learn.org/stable/install.html):
# pip install -U scikit-learn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn import metrics
from sklearn.externals import joblib
#
from sklearn.datasets import load_iris 
#
from sklearn.model_selection import train_test_split 

# Functions for running KNN, SVM, MLP, and Random Forest on data

def run_KNN(data):
    X_train = data["x_tr"]
    X_test = data["x_te"]
    y_train = data["y_tr"]
    y_test = data["y_te"]

    # training the model on training set 
    knn = KNeighborsClassifier(n_neighbors=3) 
    knn.fit(X_train, y_train) 
    
    # making predictions on the testing set 
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test) 
    
    # comparing actual response values (y_test) with predicted response values (y_pred)
    print("KNN Training Accuracy:", metrics.accuracy_score(y_train, y_pred_train)) 
    print("KNN Testing Accuracy:", metrics.accuracy_score(y_test, y_pred_test)) 
    
    # Example of making prediction for out of sample data 
    # sample = [[3, 5, 4, 2], [2, 3, 5, 4]] # make sure it is proper size
    # preds = knn.predict(sample)
    # print("Predictions:", preds) 
    
    # saving the model 
    joblib.dump(knn, 'knn_model.pkl')

    # To load model use: knn = joblib.load('knn_model.pkl')

def run_SVM(data):
    X_train = data["x_tr"]
    X_test = data["x_te"]
    y_train = data["y_tr"]
    y_test = data["y_te"]

    # training the model on training set 
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train) 
    
    # making predictions on the testing set 
    y_pred_train = svm.predict(X_train)
    y_pred_test = svm.predict(X_test) 
    
    # comparing actual response values (y_test) with predicted response values (y_pred)
    print("SVM Training Accuracy:", metrics.accuracy_score(y_train, y_pred_train)) 
    print("SVM Testing Accuracy:", metrics.accuracy_score(y_test, y_pred_test)) 
    
    # Example of making prediction for out of sample data 
    # sample = [[3, 5, 4, 2], [2, 3, 5, 4]] # make sure it is proper size
    # preds = svm.predict(sample)
    # print("Predictions:", preds) 
    
    # saving the model 
    joblib.dump(svm, 'svm_model.pkl')

    # To load model use: knn = joblib.load('svm_model.pkl')

def run_RandomForest(data):
    X_train = data["x_tr"]
    X_test = data["x_te"]
    y_train = data["y_tr"]
    y_test = data["y_te"]

    # training the model on training set 
    rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    rf.fit(X_train, y_train) 
    
    # making predictions on the testing set 
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test) 
    
    # comparing actual response values (y_test) with predicted response values (y_pred)
    print("RandomForest Training Accuracy:", metrics.accuracy_score(y_train, y_pred_train)) 
    print("RandomForest Testing Accuracy:", metrics.accuracy_score(y_test, y_pred_test)) 
    
    # Example of making prediction for out of sample data 
    # sample = [[3, 5, 4, 2], [2, 3, 5, 4]] # make sure it is proper size
    # preds = rf.predict(sample)
    # print("Predictions:", preds) 
    
    # saving the model 
    joblib.dump(rf, 'rf_model.pkl')

    # To load model use: knn = joblib.load('rf_model.pkl')

def run_MLP(data):
    X_train = data["x_tr"]
    X_test = data["x_te"]
    y_train = data["y_tr"]
    y_test = data["y_te"]

    # training the model on training set 
    mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(128,))
    mlp.fit(X_train, y_train) 
    
    # making predictions on the testing set 
    y_pred_train = mlp.predict(X_train)
    y_pred_test = mlp.predict(X_test) 
    
    # comparing actual response values (y_test) with predicted response values (y_pred)
    print("MLP Training Accuracy:", metrics.accuracy_score(y_train, y_pred_train)) 
    print("MLP Testing Accuracy:", metrics.accuracy_score(y_test, y_pred_test)) 
    
    # Example of making prediction for out of sample data 
    # sample = [[3, 5, 4, 2], [2, 3, 5, 4]] # make sure it is proper size
    # preds = mlp.predict(sample)
    # print("Predictions:", preds) 
    
    # saving the model 
    joblib.dump(mlp, 'mlp_model.pkl')

    # To load model use: knn = joblib.load('mlp_model.pkl')

# Function to get data from Firebase
def get_data(): # iris data for model testing
    iris = load_iris() 
    
    # store the feature matrix (X) and response vector (y) 
    X = iris.data 
    y = iris.target 
    
    # splitting X and y into training and testing sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    data = {"x_tr": X_train, "x_te": X_test, "y_tr": y_train, "y_te": y_test}

    return data

def get_firebase_data():
    N_PAST_TIMESTEPS = 10

    import pyrebase

    config = {
    "apiKey": "AIzaSyDYuIqP618BQ9QOscGkcLAY-OtLH2jBytk",
    "authDomain": "tomato-container-test.firebaseapp.com",
    "databaseURL": "https://tomato-container-test.firebaseio.com",
    "storageBucket": "tomato-container-test.appspot.com"
    }

    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    data_dict = db.child("data").get().val()[0]

    X_data = []
    y_data = []
    y_data_dried = []

    tomato_buffer = [[0,0,0,0] for _ in range(N_PAST_TIMESTEPS)] # [temp, humidity, is_rotten, is_dried]

    is_rotten = 0
    is_dried = 0
    for i in sorted(data_dict):
        if type(data_dict[i]) != int:
            if 'state' in data_dict[i]:
                if data_dict[i]['state'] == 'rotten':
                    is_rotten = 1

            if 'state' in data_dict[i]:
                if data_dict[i]['state'] == 'dried':
                    is_dried = 1

            tomato_buffer.append([data_dict[i]['humidity'], data_dict[i]['temperature'], is_rotten, is_dried])

            X_data.append([item for sublist in tomato_buffer[(-1 * N_PAST_TIMESTEPS):] for item in sublist])
            y_data.append(is_rotten)
            y_data_dried.append(is_dried)

            if 'batch' in data_dict[i]:
                if data_dict[i]['batch'] == True:
                    tomato_buffer = [[0,0,0,0] for _ in range(N_PAST_TIMESTEPS)] # [temp, humidity, is_rotten, is_dried]

                    is_rotten = 0
                    is_dried = 0

    # splitting X and y into training and testing sets 
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.4, random_state=1)
    # For dried tomato data: X_train, X_test, y_train, y_test = train_test_split(X_data, y_data_dried, test_size=0.4, random_state=1)
    data = {"x_tr": X_train, "x_te": X_test, "y_tr": y_train, "y_te": y_test}

    return data

def get_firebase_alt_data():
    import pyrebase

    config = {
    "apiKey": "AIzaSyDYuIqP618BQ9QOscGkcLAY-OtLH2jBytk",
    "authDomain": "tomato-container-test.firebaseapp.com",
    "databaseURL": "https://tomato-container-test.firebaseio.com",
    "storageBucket": "tomato-container-test.appspot.com"
    }

    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    data_dict = db.child("data").get().val()[0]

    X_data = []
    y_data = []

    tomato_buffer = []
    is_rotten = 0
    for i in sorted(data_dict):
        if type(data_dict[i]) != int:
            tomato_buffer.append((data_dict[i]['humidity'],data_dict[i]['temperature']))

            if 'state' in data_dict[i]:
                if data_dict[i]['state'] == 'rotten':
                    is_rotten = 1

            if 'batch' in data_dict[i]:
                if data_dict[i]['batch'] == True:
                    avg_temp = sum([d[1] for d in tomato_buffer])/len(tomato_buffer)
                    avg_humid = sum([d[0] for d in tomato_buffer])/len(tomato_buffer)
                    n_t = len(tomato_buffer)
                    datum = [avg_temp, avg_humid, n_t]
                    X_data.append(datum)
                    y_data.append(is_rotten)

                    tomato_buffer = []
                    is_rotten = 0

    # splitting X and y into training and testing sets 
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.4, random_state=1)
    data = {"x_tr": X_train, "x_te": X_test, "y_tr": y_train, "y_te": y_test}

    return data

# Iris testing data
print("#"*90)
print("#"*90)
print("\nIRIS DATA INITIAL TESTS *IGNORE*\n")
print("#"*90)
print("#"*90)
print()


data = get_data()

print("\nK NEAREST NEIGHBOR:\n")

run_KNN(data)

print()
print("#"*30)
print("\nSUPPORT VECTOR MACHINE:\n")

run_SVM(data)

print()
print("#"*30)
print("\nRANDOM FOREST:\n")

run_RandomForest(data)

print()
print("#"*30)
print("\nMULTILAYER PERCEPTRON:\n")

run_MLP(data)

print()

# Prediction on past N days of data
print("#"*90)
print("#"*90)
print("\nTEST 1 - Use Past N (N=10) Days to Predict Tomato Health\n")
print("#"*90)
print("#"*90)
print()

data = get_firebase_data()

print("\nK NEAREST NEIGHBOR:\n")

run_KNN(data)

print()
print("#"*30)
print("\nSUPPORT VECTOR MACHINE:\n")

run_SVM(data)

print()
print("#"*30)
print("\nRANDOM FOREST:\n")

run_RandomForest(data)

print()
print("#"*30)
print("\nMULTILAYER PERCEPTRON:\n")

run_MLP(data)

print()


# ***Important Note*** the saved model data will overwrite everytime you run the same model functions

# Prediction based on average temperature, average humidity, and number of days
print("#"*90)
print("#"*90)
print("\nTEST 2 - Use Average Temp, Average Humidity, and # of Days to Predict Tomato Health\n")
print("#"*90)
print("#"*90)
print()

data = get_firebase_alt_data()

print("\nK NEAREST NEIGHBOR:\n")

run_KNN(data)

print()
print("#"*30)
print("\nSUPPORT VECTOR MACHINE:\n")

run_SVM(data)

print()
print("#"*30)
print("\nRANDOM FOREST:\n")

run_RandomForest(data)

print()
print("#"*30)
print("\nMULTILAYER PERCEPTRON:\n")

run_MLP(data)

print()

# TODO Potential More Complex Model RNN/LSTM