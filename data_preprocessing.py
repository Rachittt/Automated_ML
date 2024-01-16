from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(df, target_column):
    # Identify numerical and categorical columns
    numerical_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Handle missing values in numerical columns
    numerical_imputer = SimpleImputer(strategy='mean')
    df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])

    # Handle missing values in categorical columns
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

    # Encode categorical variables
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])

    # Split the data into features (X) and target variable (y)
    X = df.drop(target_column, axis=1)  # Adjust 'target_column' based on your dataset
    y = df[target_column]

    df.to_csv('download/preprocessed_data.csv', index=False)
    return df, X, y


def scale_encode_split(X, y):
    # Separate numerical and categorical columns
    numerical_columns = X.select_dtypes(include=['number']).columns
    categorical_columns = X.select_dtypes(include=['object']).columns

    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    # Encode categorical features
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        X[column] = label_encoder.fit_transform(X[column])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    return X_train, X_test, y_train, y_test


def save_preprocessed_data(X_train, X_test, y_train, y_test):
    # Save the preprocessed data to files or databases for future use
    X_train.to_csv('download/X_train.csv', index=False)
    X_test.to_csv('download/X_test.csv', index=False)
    y_train.to_csv('download/y_train.csv', index=False)
    y_test.to_csv('download/y_test.csv', index=False)