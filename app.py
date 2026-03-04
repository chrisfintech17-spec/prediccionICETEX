import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load the pre-trained SVR model
svr_model = joblib.load('svr_model.joblib')

# Load the encoded feature columns list
encoder_feature_columns_v2 = joblib.load('encoded_feature_columns.joblib')

def main():
    st.title("Prediction App")
    st.write("This app predicts based on user inputs.")

    # 1. Infer original categorical columns and their unique categories from encoder_feature_columns_v2
    input_categorical_features = {}
    inferred_original_cols_from_encoder = []
    categories_map_for_ohe = {}

    for col_name in encoder_feature_columns_v2:
        if '_' in col_name:
            original_feature, category_value = col_name.rsplit('_', 1)
            if original_feature not in categories_map_for_ohe:
                categories_map_for_ohe[original_feature] = []
                inferred_original_cols_from_encoder.append(original_feature) # Keep order for OHE categories
            categories_map_for_ohe[original_feature].append(category_value)

    # Sort categories within each feature for consistent OHE initialization
    for feature in categories_map_for_ohe:
        categories_map_for_ohe[feature] = sorted(list(set(categories_map_for_ohe[feature])))
    
    ohe_categories_list_for_input = [categories_map_for_ohe[col] for col in inferred_original_cols_from_encoder]
    
    # Manually add 'NIVEL DE FORMACIÓN' categories as it was not part of the one-hot encoded features 
    # but is a required input feature. These categories were derived from the kernel state variable 'categories'.
    # Ensure it's added to `input_categorical_features` to populate the selectbox.
    input_categorical_features = {k: v for k, v in categories_map_for_ohe.items()}
    input_categorical_features['NIVEL DE FORMACIÓN'] = ['Formación técnica profesional', 'Tecnológico', 'Universitario', 'Exterior']
    
    # Define the order of features for display in Streamlit and for creating the input DataFrame
    ordered_features = [
        'ESTRATO SOCIOECONÓMICO',
        'CATEGORÍA DEL MUNICIPIO DE ORIGEN',
        'SECTOR IES',
        'MODALIDAD DE LÍNEA',
        'MODALIDAD DEL CRÉDITO',
        'RANGO DEL VALOR TOTAL DESEMBOLSADO',
        'NIVEL DE FORMACIÓN'
    ]

    # Create input widgets for each categorical feature
    user_inputs = {}
    st.sidebar.header("User Input Features")
    for feature in ordered_features:
        options = input_categorical_features.get(feature, [])
        if options:
            user_inputs[feature] = st.sidebar.selectbox(f"Select {feature}", options)
        else:
            st.sidebar.warning(f"No categories found for feature: {feature}")
            user_inputs[feature] = st.sidebar.text_input(f"Enter {feature}", "")

    st.write("### User Inputs")
    st.write(user_inputs)

    if st.sidebar.button('Predict'):
        # Convert user_inputs to a DataFrame for encoding
        df_user_input = pd.DataFrame([user_inputs])

        # Initialize OneHotEncoder with explicit categories and handle unknown values gracefully
        # It's important to use the categories learned from the training data for consistent encoding.
        one_hot_encoder_for_prediction = OneHotEncoder(categories=ohe_categories_list_for_input, handle_unknown='ignore', sparse_output=False)

        # Identify columns in df_user_input to be encoded (these are the columns that were part of the original OHE training)
        cols_to_encode_from_user_input = [col for col in inferred_original_cols_from_encoder if col in df_user_input.columns]

        if not cols_to_encode_from_user_input:
            st.error("No matching categorical columns found in user input to apply encoding.")
            return

        # Transform the identified categorical data from df_user_input
        data_to_encode_user_input = df_user_input[cols_to_encode_from_user_input]
        encoded_data_array_user_input = one_hot_encoder_for_prediction.fit_transform(data_to_encode_user_input)

        # Create a temporary DataFrame for the newly encoded features from user input
        generated_feature_names_user_input = one_hot_encoder_for_prediction.get_feature_names_out(cols_to_encode_from_user_input)
        df_encoded_temp_user_input = pd.DataFrame(encoded_data_array_user_input, columns=generated_feature_names_user_input, index=df_user_input.index)

        # Create a final DataFrame for the encoded features, initialized with zeros,
        # and then fill it with values from `df_encoded_temp_user_input`. This ensures all expected
        # columns from `encoder_feature_columns_v2` are present and in the correct order.
        df_final_encoded_features_user_input = pd.DataFrame(0, index=df_user_input.index, columns=encoder_feature_columns_v2)
        for col in df_encoded_temp_user_input.columns:
            if col in df_final_encoded_features_user_input.columns:
                df_final_encoded_features_user_input[col] = df_encoded_temp_user_input[col]

        # Drop the 'NIVEL DE FORMACIÓN' column as it was not seen during model training
        # (it was dropped in the previous notebook step before prediction)
        # This column is not present in df_final_encoded_features_user_input because it was not OHE. 
        # It needs to be handled separately if it was concatenated, but in this flow, it's not present here.

        df_processed_for_prediction_user_input = df_final_encoded_features_user_input

        try:
            prediction = svr_model.predict(df_processed_for_prediction_user_input)
            st.write("### Prediction:")
            st.write(f"The predicted value is: {prediction[0]:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.write(df_processed_for_prediction_user_input.columns.tolist())
            st.write(f"Expected columns from model: {svr_model.feature_names_in_}")

if __name__ == '__main__':
    main()
