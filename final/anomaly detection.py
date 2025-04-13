# anomaly_detection.py
import torch
import pandas as pd
import numpy as np
from anomaly_model import AnomalyAttention, AnomalyTransformer  # Import model definition
from sklearn.cluster import KMeans
# Load pre-trained model
def load_model():
    # Assuming you have the same input shape as during training
    df_x = pd.read_csv('test.txt')
    df_y=pd.read_csv('ytest.txt')
    # df.drop('Is Anomaly',axis=1,inplace=True)
    # df.drop('Date',axis=1,inplace=True)
    

    X = df_x.values
    N = X.shape[0]
    d_model = 512
    
    model = AnomalyTransformer(N, d_model, hidden_dim=64)
    model.load_state_dict(torch.load('anomaly_transformer_weights.pth'))
    model.eval()  # Set to evaluation mode
    return model

# Anomaly score calculation
def calculate_anomaly_score(model, new_data):
    new_data_tensor = torch.FloatTensor(new_data)
    
    with torch.no_grad():  # Disable gradient calculation during inference
        anomaly_scores = model.anomaly_score(new_data_tensor)
    
    return anomaly_scores

# Load new data for anomaly detection
new_data = pd.read_csv('test.txt') 
# new_data.drop('Date',axis=1,inplace=True)


# new_data.drop('Is Anomaly',axis=1,inplace=True)
# new_data.drop('Date',axis=1,inplace=True)
new_data_values = new_data.values  # Convert to numpy array
padded_values=np.pad(new_data_values,((0,0),(0,512-new_data_values.shape[1])),mode="constant")
# Load the model
model = load_model()

# Calculate anomaly scores
anomaly_scores = calculate_anomaly_score(model, padded_values)
anomaly_scores=anomaly_scores.reshape(-1,1)

max=torch.max(anomaly_scores)
min=torch.min(anomaly_scores)

norm_anomaly_scores=0.1+0.8*((anomaly_scores-min)/(max-min))
# Output the anomaly scores
kmeans=KMeans(n_clusters=2,random_state=42)
kmeans.fit(norm_anomaly_scores)
centers=kmeans.cluster_centers_.flatten()
threshold=np.mean(centers)
predic_output=[]
for i in range(len(norm_anomaly_scores)):
    if norm_anomaly_scores[i]>threshold:
        print(f"{i+2} : {norm_anomaly_scores[i]}")
        predic_output.append(1)
    else:
        predic_output.append(0)

print(predic_output)


