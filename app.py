st.title('Protein Interaction Predictor')

st.write("""
Input the protein sequence to predict interactions.
""")

# Input fields for protein features
sequence1 = st.text_input('sequence 1')
sequence2 = st.text_input('sequence 2')
# Add more features as needed



if st.button('Predict'):
      
    prediction, prediction_proba = predict_interaction(sequence1, sequence2)
    if prediction[0] == 1:
            st.write('The given sequnces doesnot interact to each other.')
    else:
            st.write('The given sequnces have interaction')
            st.write(f'Prediction score: {prediction_proba[0][prediction[0]]:.2f}')

    
