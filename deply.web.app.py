import numpy as np
import pickle
import streamlit as st

load_model=pickle.load(open(r"C:\Users\elvin\OneDrive\Documents\Data Science\Stats and ML\ML\trained_model.sav",'rb'))


def iris_prediction(input_data):
    
    
    input_as_numpy = np.asarray(input_data)
    
    input_reshape = input_as_numpy.reshape(1,-1)
    
    pred= load_model.predict(input_reshape)
    print(pred)
    
    if (pred[0]==0):
        return'The flower is setosa'
    elif pred[0] == 1:
        return'The flower is versicolor'
    else:
        return'The flower is virginica'
        

def main():
      
    st.title("Iris Species Prediction App")
    st.write("Enter the details below to predict the species of an Iris flower:")

    sepal_length = st.text_input("Sepal Length")
    sepal_width = st.text_input("Sepal Width")
    petal_length = st.text_input("Petal Length")
    petal_width = st.text_input("Petal Width")

    input_data = (sepal_length, sepal_width, petal_length, petal_width)

    diagnosis=''
    if st.button('Iris Test Result'):
        diagnosis=iris_prediction(input_data)
    st.success(diagnosis)
    
    
if __name__=='__main__':
    main()
    
    

    