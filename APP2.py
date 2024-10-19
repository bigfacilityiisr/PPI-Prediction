import streamlit as st
import joblib
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
import requests

# Load your trained model
model = joblib.load('Best Model.pkl')
top_n_features = np.load('top_n_features.npy')

def calculate_aac(sequence):
    protein_analysis = ProteinAnalysis(sequence)
    aac = protein_analysis.get_amino_acids_percent()
    return aac

def calculate_k_spaced_pairs(sequence, k):
    pairs = {}
    length = len(sequence)
    for i in range(length - k):
        pair = sequence[i] + sequence[i + k]
        pairs[pair] = pairs.get(pair, 0) + 1
    return pairs

def calculate_pse_aac_type_ii(sequence, k, max_length):
    aac = calculate_aac(sequence)
    pairs = calculate_k_spaced_pairs(sequence, k)
    dipeptides = [sequence[i:i + 2] for i in range(len(sequence) - 1)]
    dipeptide_counts = {dipeptide: dipeptides.count(dipeptide) for dipeptide in set(dipeptides)}
    pse_aac = list(aac.values())
    pse_aac.extend(pairs.values())
    pse_aac.extend(dipeptide_counts.values())
    if len(pse_aac) < max_length:
        pse_aac.extend([0] * (max_length - len(pse_aac)))
    elif len(pse_aac) > max_length:
        pse_aac = pse_aac[:max_length]
    return np.array(pse_aac)

# Preprocess a single input sequence
def preprocess_sequence(sequence1, sequence2, k=3, max_length=3601):
    pse_aac_1 = calculate_pse_aac_type_ii(sequence1, k, max_length)
    pse_aac_2 = calculate_pse_aac_type_ii(sequence2, k, max_length)
    combined_features = np.concatenate((pse_aac_1, pse_aac_2))
    return combined_features

# Predict function for a single input
def predict_interaction(sequence1, sequence2):
    input_features = preprocess_sequence(sequence1, sequence2)
    input_features = input_features.reshape(1, -1)
    input_features_s = input_features[:, top_n_features] 
    
    prediction = model.predict(input_features_s)
    prediction_proba = model.predict_proba(input_features_s)
    
    return prediction, prediction_proba

# Function to get protein sequence by UniProt ID
def get_protein_sequence(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    
    if response.status_code == 200:
        fasta_lines = response.text.strip().split('\n')
        sequence = ''.join(fasta_lines[1:])  # Combine the sequence lines
        return sequence
    else:
        return None

# CSS for styling
st.markdown("""
<style>
    .button {
        background-color: #4CAF50; /* Green */
        border: none;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
        transition: background-color 0.3s;
    }
    .button:hover {
        background-color: #45a049; /* Darker green */
    }
    .stButton {
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("About")
st.sidebar.write("""
This application allows you to input protein sequences and predict their interaction.
""")   

st.title('Protein Interaction Predictor')

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Input Sequences", "Retrieve Sequences"])

# Tab 1: Input Sequences
with tab1:
    st.write("### Input your protein sequences directly:")
    
    sequence1 = st.text_area('Sequence 1', height=100)
    sequence2 = st.text_area('Sequence 2', height=100)

    if st.button('Predict Interaction', key='predict'):
        if sequence1 and sequence2:
            prediction, prediction_proba = predict_interaction(sequence1, sequence2)
            if prediction[0] == 1:
                st.write('The given sequences do not interact with each other.')
            else:
                st.write('The given sequences have an interaction.')
                st.write(f'Prediction score: {prediction_proba[0][prediction[0]]:.2f}')
        else:
            st.error("Please enter both sequences.")

# Tab 2: Retrieve Sequences
with tab2:
    st.write("### Retrieve protein sequences using UniProt IDs:")
    
    uniprot_id_1 = st.text_input('UniProt ID for Sequence 1')
    uniprot_id_2 = st.text_input('UniProt ID for Sequence 2')

    if st.button('Retrieve Sequences', key='retrieve'):
        sequence1 = get_protein_sequence(uniprot_id_1)
        sequence2 = get_protein_sequence(uniprot_id_2)
        
        if sequence1 and sequence2:
            st.success('Sequences retrieved successfully!')
            st.write(f'Sequence 1: {sequence1}')
            st.write(f'Sequence 2: {sequence2}')
        else:
            st.error('Error retrieving sequences. Please check the UniProt IDs.')

    # Add a button to predict after retrieval
    if 'sequence1' in locals() and 'sequence2' in locals():
        if st.button('Predict Interaction After Retrieval', key='predict_after'):
            prediction, prediction_proba = predict_interaction(sequence1, sequence2)
            if prediction[0] == 1:
                st.write('The given sequences do not interact with each other.')
            else:
                st.write('The given sequences have an interaction.')
                st.write(f'Prediction score: {prediction_proba[0][prediction[0]]:.2f}')
