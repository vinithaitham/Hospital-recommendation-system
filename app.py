from flask import Flask, request, jsonify, render_template
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load dataset
df = pd.read_csv("andhra_hospitals_full.csv")

# Load trained LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        return self.fc(lstm_out[:, -1, :])

input_dim = 5  # Adjust according to features
hidden_dim = 64
output_dim = len(df)

lstm_model = LSTMModel(input_dim, hidden_dim, output_dim)
lstm_model.load_state_dict(torch.load("lstm_model.pth"))
lstm_model.eval()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend_hospitals():
    data = request.json
    surgery = data["surgery"]
    location = data["location"]

    # Filter dataset based on user input
    filtered_df = df[(df["Surgery/Treatment Name"] == surgery) & 
                     (df["Locality"] == location)]

    # Group hospitals, keeping the highest rating and lowest cost
    filtered_df = (
        filtered_df.groupby("Hospital Name", as_index=False)
        .agg({"Locality": "first", "Hospital Ratings": "max", "Charges": "min"})
        .sort_values(by=["Hospital Ratings"], ascending=False)  # Sort by ratings
    )

    return jsonify(filtered_df.head(5).to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)
