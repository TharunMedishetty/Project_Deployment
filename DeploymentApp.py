import pickle
import streamlit as st
import scikit-learn
 
# loading the trained model
pickle_in = open('Regression_Model.pkl', 'rb') 
Regression_Model = pickle.load(pickle_in)
 
@st.cache()

# defining the function which will make the prediction using the data which the user inputs 
def prediction(name, item_condition_id, category_name, brand_name, shipping, item_description):   
 
    # Making predictions 
    prediction = Regression_Model.predict( 
        [[name, item_condition_id, category_name, brand_name, shipping, item_description]])
        
        
# this is the main function in which we define our webpage  
def main():       
   # front end elements of the web page 
   html_temp = """ 
   <div style ="background-color:blue;padding:13px"> 
   <h1 style ="color:black;text-align:center;">Mercari Price Suggestion ML App</h1> 
   </div> 
     """
      
   # display the front end aspect
   st.markdown(html_temp, unsafe_allow_html = True) 
      
   # following lines create boxes in which user can enter data required to make prediction 
   name = st.text('name')
   item_condition_id = st.selectbox('item_condition_id',(1,2,3,4,5))
   category_name = st.text('category_name')
   brand_name = st.text('brand_name')
   shipping = st.selectbox('shipping',(1,0))
   item_description = st.text('item_description')
   result =""
      
   # when 'Predict' is clicked, make the prediction and store it 
   if st.button("Predict"): 
       result = prediction(name, item_condition_id, category_name, brand_name, shipping, item_description) 
       st.success('Price for the product is {}'.format(result))
        
     
if __name__=='__main__': 
    main()
  
