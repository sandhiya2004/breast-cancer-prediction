import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
import pickle


def create_model(data): 
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    # Feature selection (removes low-variance features)
    selector = VarianceThreshold(threshold=0.01)
    X_selected = selector.fit_transform(X)
    
    # Handling class imbalance using SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_selected, y)

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)
    
    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Test model
    y_pred = model.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    
    return model, scaler, selector


def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
    return data


def main():
    data = get_clean_data()
    model, scaler, selector = create_model(data)
    
    # Save models
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('model/selector.pkl', 'wb') as f:
        pickle.dump(selector, f)


if __name__ == '__main__':
    main()
