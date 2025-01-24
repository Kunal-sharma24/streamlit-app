import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Title of the app
st.title('Enhanced Streamlit App with Numpy, Matplotlib, and Pandas')

# Input fields
name = st.text_input('Enter your name:')
age = st.number_input('Enter your age:', min_value=0, max_value=120)

# Button to submit
if st.button('Submit'):
    st.write(f'Hello, {name}! You are {age} years old.')

# Display a slider
slider_value = st.slider('Select a value:', 0, 100, 50)
st.write(f'Slider value: {slider_value}')

# Generate some random data using numpy
data = np.random.randn(100)

# Create a pandas DataFrame
df = pd.DataFrame(data, columns=['Random Data'])

# Display the DataFrame
st.write('Random Data:', df)

# Plot the data using matplotlib
fig, ax = plt.subplots()
ax.hist(df['Random Data'], bins=20)
st.pyplot(fig)
