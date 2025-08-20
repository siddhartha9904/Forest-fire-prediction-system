import streamlit as st
import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler

with open("forest.pkl","rb") as f:
    model=pkl.load(f)

with open("scaler.pkl","rb") as f:
    ss=pkl.load(f)

st.title("Forest Fire Prediction System")

st.markdown("Enter following environment condition values:")

temp=st.number_input("Temperature",min_value=0.0,max_value=50.0,step=0.1)
wind=st.number_input("wind speed(km/h)",min_value=0.0,max_value=50.0,step=0.1)
humidity=st.slider("Humidity",min_value=0.0,max_value=100.0,step=0.1)
rain=st.number_input("rain mm",min_value=0.0,max_value=20.0,step=0.1)
ffmc=st.number_input("FFMC",min_value=0.0,max_value=100.0,step=0.1)
dmc=st.number_input("DMC",min_value=0.0,max_value=300.0,step=0.1)
month=st.selectbox("MONTH",list(range(1,13)))
day=st.selectbox("day",list(range(1,8)))

if st.button("predict"):
    input_data=np.array([[day,month,temp,wind,humidity,rain,ffmc,dmc,month,day]])
    input_data.reshape(1,-1)
    scaled=ss.transform(input_data)
    y_pred=model.predict(input_data)
    print(y_pred)