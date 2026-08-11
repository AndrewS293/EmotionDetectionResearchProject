import numpy as np
from sklearn.model_selection import train_test_split

'''
These are all .py files that perform different functions and are imported into this main file to be used
'''

from pipelines.data_loader import load_all_subjects as load_all_wesad
from pipelines.amigos_loader import load_all_subjects as load_all_amigos
from pipelines.feature_extraction import (
    create_model_windows,
    #create_raw_windows,

    build_logistic_features,
    build_svm_features,
    build_knn_features,
    build_rf_features,
    build_gb_features,
    build_xgb_features,
    build_amigos_features
)
from pipelines.evaluate import evaluate_model
from pipelines.logistics_pipeline import build_model as logistic_model
from pipelines.svm_pipeline import build_model as svm_model
from pipelines.randomforest_pipeline import build_model as rf_model
from pipelines.xgboost_pipeline import build_model as xgb_model
from pipelines.knn_pipeline import build_model as knn_model
from pipelines.gradientboost_pipeline import build_model as gb_model

def train_models(DATA_DIR):

    if "AMIGOS" in DATA_DIR:    
        #gets all the data from all the subjects
        signals, labels = load_all_amigos(DATA_DIR)

    else:
        signals, labels = load_all_wesad(DATA_DIR)

    #print(signals)
    #print (labels)
    
    feature_builders = {
        'Logistic Regression': build_logistic_features,
        'SVM': build_svm_features,
        'KNN': build_knn_features,
        'Random Forest': build_rf_features,
        'Gradient Boosting': build_gb_features,
        'XGBoost': build_xgb_features
    }
    
    
    #all the different models/algorithms to be used, each using a method imported earlier
    models = {
        'Logistic Regression': logistic_model(),
        'SVM': svm_model(),
        'KNN': knn_model(),
        'Random Forest': rf_model(),
        'Gradient Boosting': gb_model(),
        'XGBoost': xgb_model()
    }
    results = {}
    
    
    #for each different model it is trained and tested then it feeds into the evaluate model method
    X_models = {}
    y_models = {}
    model_feature_names = {}
    
    for name, model in models.items():
    
        print(f"\n===== TRAINING {name} =====")


        if "AMIGOS" in DATA_DIR:    
            print ("Actually worked")
            builder = build_amigos_features
            #builder = feature_builders['KNN']
            #print(builder)
        else:




            
            builder = feature_builders[name]
    
        all_X = []
        all_y = []

        #print(f"Subjects: {len(signals)}")
        # BUILD FEATURES
        for i, (signal, label) in enumerate(zip(signals, labels)):

            

            #print(f"Processing subject {i+1}/{len(signals)}")
            #print("Signal shape:", signal.shape)
            #print("Label shape:", label.shape)
    
            Xi, yi, feature_names = create_model_windows(
                signal,
                label,
                builder
            )
            #print(feature_names)
    
            #print("Windows created:", len(Xi))
            
            all_X.extend(Xi)
            all_y.extend(yi)


        X_models[name] = all_X
        y_models[name] = all_y
        model_feature_names[name] = feature_names
    
        all_X = np.array(all_X)
        all_y = np.array(all_y)
    
        print("Dataset Shape:", all_X.shape)
    
        X_train, X_test, y_train, y_test = train_test_split(
            all_X,
            all_y,
            test_size=0.2,
            random_state=42,
            stratify=all_y
        )
    
        model.fit(X_train, y_train)
    
        metrics = evaluate_model(
            model,
            X_test,
            y_test,
            name
        )
    
        results[name] = metrics
        
    
    #this prints the results of each
    print('\n===== FINAL RESULTS =====')
    
    for k, v in results.items():
        print(
            f"{k} | Accuracy: {v['accuracy']:.4f} | "
            f"F1: {v['f1']:.4f}"
        )

    return models, signals, labels, X_models, y_models, model_feature_names, all_X, all_y
    
