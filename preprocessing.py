import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def preprocess_restaurant_data(file_path):
    print("Starting data preprocessing...")
    
    # 1. Load the data
    print("\nLoading dataset...")
    df = pd.read_csv(file_path)
    initial_shape = df.shape
    print(f"Initial dataset shape: {initial_shape}")

    # 2. Remove unnecessary columns
    print("\nRemoving unnecessary columns...")
    columns_to_drop = ['opentable_support', 'photo_count', 'area', 'votes']
    df = df.drop(columns=columns_to_drop)
    
    # 3. Handle missing values
    print("\nHandling missing values...")
    print("Missing values before cleaning:")
    print(df.isnull().sum())
    
    # Fill missing values appropriately
    df['cuisines'] = df['cuisines'].fillna('Not Specified')
    df['timings'] = df['timings'].fillna('Not Specified')
    df['highlights'] = df['highlights'].fillna('None')
    
    # Drop rows where critical information is missing
    critical_columns = ['name', 'city', 'latitude', 'longitude', 'aggregate_rating', 'price_range']
    df = df.dropna(subset=critical_columns)
    
    # 4. Remove duplicates
    print("\nRemoving duplicates...")
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['name', 'latitude', 'longitude'])
    duplicates_removed = initial_rows - len(df)
    print(f"Removed {duplicates_removed} duplicate entries")

    # 5. Clean and standardize text data
    print("\nCleaning text data...")
    df['name'] = df['name'].str.strip()
    df['city'] = df['city'].str.strip()
    df['locality'] = df['locality'].str.strip()
    
    # 6. Extract features from highlights
    print("\nExtracting features from highlights...")
    # Convert string representation of list to actual list
    df['highlights'] = df['highlights'].apply(lambda x: str(x).strip('[]').replace("'", "").split(', '))
    
    # Create binary columns for common highlights
    common_highlights = ['Delivery', 'Takeaway Available', 'Credit Card', 'Indoor Seating', 
                        'Air Conditioned', 'Outdoor Seating', 'WiFi']
    
    for highlight in common_highlights:
        df[f'has_{highlight.lower().replace(" ", "_")}'] = df['highlights'].apply(
            lambda x: 1 if highlight in x else 0)

    # 7. Encode categorical variables
    print("\nEncoding categorical variables...")
    # Label encode city and locality
    le_city = LabelEncoder()
    le_locality = LabelEncoder()
    
    df['city_encoded'] = le_city.fit_transform(df['city'])
    df['locality_encoded'] = le_locality.fit_transform(df['locality'])
    
    # Save encoding mappings
    city_mapping = dict(zip(le_city.classes_, le_city.transform(le_city.classes_)))
    locality_mapping = dict(zip(le_locality.classes_, le_locality.transform(le_locality.classes_)))
    
    # 8. Feature scaling
    print("\nPerforming feature scaling...")
    scaler = MinMaxScaler()
    
    # Scale numerical features
    numerical_features = ['average_cost_for_two', 'price_range', 'aggregate_rating']
    df[['scaled_cost', 'scaled_price_range', 'scaled_rating']] = scaler.fit_transform(df[numerical_features])
    
    # 9. Create cuisine count feature
    df['cuisine_count'] = df['cuisines'].str.count(',') + 1
    
    # 10. Save preprocessing mappings
    preprocessing_info = {
        'city_mapping': city_mapping,
        'locality_mapping': locality_mapping,
        'scaler': scaler,
        'numerical_features': numerical_features
    }
    
    # 11. Final cleaning
    # Remove any remaining rows with invalid coordinates
    df = df[df['latitude'].between(6, 37) & df['longitude'].between(68, 98)]  # India's geographical boundaries
    
    print("\nPreprocessing complete!")
    print(f"Final dataset shape: {df.shape}")
    print(f"Total rows removed: {initial_shape[0] - df.shape[0]}")
    
    # Save preprocessed data
    df.to_csv('preprocessed_restaurants.csv', index=False)
    print("\nPreprocessed data saved to 'preprocessed_restaurants.csv'")
    
    # Print data quality report
    print("\n=== Data Quality Report ===")
    print(f"Total restaurants: {len(df)}")
    print(f"Cities covered: {len(df['city'].unique())}")
    print(f"Average rating: {df['aggregate_rating'].mean():.2f}")
    print(f"Price range distribution:\n{df['price_range'].value_counts().sort_index()}")
    print("\nMissing values in final dataset:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    return df, preprocessing_info

# Execute preprocessing
if __name__ == "__main__":
    try:
        df, preprocessing_info = preprocess_restaurant_data('restaurants.csv')
        print("\nYou can now use the preprocessed data for further analysis!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")