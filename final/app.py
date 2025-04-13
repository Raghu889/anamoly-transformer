import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import torch
from anomaly_model import AnomalyAttention, AnomalyTransformer 
# Title
import matplotlib.pyplot as plt
st.title("Anomaly Detection App with Column Selection")

# Sidebar instructions
st.sidebar.header("Upload a CSV File")
st.sidebar.write("Make sure the file has numerical data for anomaly detection.")

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    
    # Display the data
    st.subheader("Uploaded Data")
    st.dataframe(data)

    # Allow the user to select columns for anomaly detection
    st.sidebar.header("Select Columns for Anomaly Detection")
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_columns:
        



        selected_columns = st.sidebar.multiselect(
            "Select the columns to use for anomaly detection:",
            numeric_columns,
            default=numeric_columns
        )

        if selected_columns:
            st.subheader("Multi-Line Graph")

            fig, ax = plt.subplots(figsize=(10, 6))

            for column in selected_columns:
                ax.plot(data.index, data[column], label=column)  # Plot each column

            ax.set_xlabel("Index", fontsize=12)
            ax.set_ylabel("Values", fontsize=12)
            ax.set_title("Multi-Line Graph", fontsize=16)
            ax.legend(title="Columns", loc="best", fontsize=10)
            ax.grid(True)

            st.pyplot(fig)









            st.subheader("Selected Data for Anomaly Detection")
            selected_data = data[selected_columns]
            st.write(selected_data)

            def load_model():
                
                

                X = selected_data.values
                N = X.shape[0]
                d_model = 512

                model = AnomalyTransformer(N, d_model, hidden_dim=64)
                model.load_state_dict(torch.load('anomaly_transformer_weights.pth'))
                model.eval()  # Set to evaluation mode
                return model
            
            def calculate_anomaly_score(model, new_data):
                new_data_tensor = torch.FloatTensor(new_data)

                with torch.no_grad():  # Disable gradient calculation during inference
                    anomaly_scores = model.anomaly_score(new_data_tensor)

                return anomaly_scores
            
            

           
            # Load the model
            model = load_model()
            new_data_values=selected_data.values
            # Calculate anomaly scores
            X_padded=np.pad(new_data_values,((0,0),(0,512-new_data_values.shape[1])),mode='constant')
           
            anomaly_scores = calculate_anomaly_score(model, X_padded)


            max=torch.max(anomaly_scores)
            min=torch.min(anomaly_scores)

            log_anomaly_scores = np.log1p(anomaly_scores.numpy())  # Use log1p for numerical stability

            # Normalize log-transformed scores
            min_log_score = np.min(log_anomaly_scores)
            max_log_score = np.max(log_anomaly_scores)
            norm_anomaly_scores = (log_anomaly_scores - min_log_score) / (max_log_score - min_log_score)

            # Convert back to PyTorch tensor
            norm_anomaly_scores = torch.from_numpy(norm_anomaly_scores)
            st.sidebar.header("set anomaly threshold")
            threshold = st.sidebar.slider(
                "Select the anomaly threshold:",
                min_value=float(f"{norm_anomaly_scores.min():.5f}"),
                max_value=float(f"{norm_anomaly_scores.max():.5f}"),
                step=0.00001
            )
            # Output the anomaly scores
            # st.subheader("anomaly points")
            anomalies=[]
            # for i in range(len(norm_anomaly_scores)):
            #   if norm_anomaly_scores[i]>threshold:
            #     st.write(f"{i} | {norm_anomaly_scores[i]}")

            anomalies = [i for i in range(len(norm_anomaly_scores)) if norm_anomaly_scores[i] > threshold]

            fig, ax = plt.subplots(figsize=(10, 6))

            for column in selected_columns:
                ax.plot(data.index, data[column], label=column)  # Plot each column

            # Highlight anomalies
            for column in selected_columns:
                ax.scatter(
                    data.index[anomalies],
                    data[column].iloc[anomalies],
                    color='red',
                    label=f"Anomalies in {column}",
                    zorder=5
                )

            ax.set_xlabel("Index", fontsize=12)
            ax.set_ylabel("Values", fontsize=12)
            ax.set_title("Multi-Line Graph with Anomalies", fontsize=16)
            # ax.legend(title="Columns", loc="best", fontsize=5)
            ax.grid(True)

            st.pyplot(fig)

            # Display anomaly details
            st.subheader("Anomaly Points")
            for anomaly_idx in anomalies:
                st.write(f"Index: {anomaly_idx}, Anomaly Score: {norm_anomaly_scores[anomaly_idx]:.5f}")

            


            
        else:
            st.warning("Please select at least one column for anomaly detection.")
    else:
        st.error("The uploaded file does not contain any numerical columns. Please upload a valid dataset.")
else:
    st.warning("Please upload a CSV file to proceed.")

