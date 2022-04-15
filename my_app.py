# streamlit run C:\Users\HP\.spyder-py3\my_app.py

import pandas as pd
import streamlit as st
import plotly_express as px
#from ipywidgets import interact



header=st.container()
dataset=st.container()
sum_stat=st.container()
col1, col2, col3,col4,col5,col6 = st.columns(6)

graph=st.container()

input=st.container()
model_training=st.container()


 

with header:
    st.title('Crops with their soil & environmental conditions')
    st.text('The purpose of this project is the present a data on various crops with their')
    st.text('environmental and soil conditions. Based on this dataset, a machine learning') 
    st.text('model was developed to predict the most suitable crop given the conditions') 
    st.text('specified by the user')
    
    
    

with dataset:
    st.header('Dataset')
    #st.text('Click the following link to download the dataset used in this tutorial')

    url = 'https://drive.google.com/file/d/1kGYrUUF-jLf_9wif5IZxGUsOY4Ny5DDr/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    data = pd.read_csv(path)
    data.rename(columns={"N":"Nitrogen","P":"Phosphorus","K":"Potassium"},inplace=True)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    if st.checkbox("Show the Data"):
        st.write(data)
    #st.download_button("Download Data", data)
    st.download_button(label="Download data as CSV", data=data.to_csv(), file_name='large_df.csv', mime='text/csv')
    st.subheader("SUMMARY OF CONDITIONS FOR SELECTED CROP")
    crops=list(data['label'].value_counts().index)
with sum_stat:   
    crops_menu = st.selectbox('Select crop name to display summary statistics:',(crops))
    if st.checkbox("Show the Summary Statistics for the selected crops",(crops_menu)):
        with col1:
            st.text(".................................")
            st.text(("NITROGEN"))
            st.text(".................................")
            st.write("Maximum = ", round(data[data['label']==crops_menu]["Nitrogen"].max(),2))
            st.write("Minimum = ", round(data[data['label']==crops_menu]["Nitrogen"].min(),2))
            st.write("Average = ", round(data[data['label']==crops_menu]["Nitrogen"].mean(),2))
        
        with col2:
            st.text(".................................")
            st.text("PHOSPHORUS")
            st.text(".................................")
            st.write("Maximum = ", round(data[data['label']==crops_menu]["Phosphorus"].max(),2))
            st.write("Minimum = ", round(data[data['label']==crops_menu]["Phosphorus"].min(),2))
            st.write("Average = ", round(data[data['label']==crops_menu]["Phosphorus"].mean(),2))
        with col3:
            st.text(".................................")
            st.text("POTASSIUM")
            st.text(".................................")
            st.write("Maximum = ", round(data[data['label']==crops_menu]["Potassium"].max(),2))
            st.write("Minimum = ", round(data[data['label']==crops_menu]["Potassium"].min(),2))
            st.write("Average = ", round(data[data['label']==crops_menu]["Potassium"].mean(),2))
        with col4:
            st.text(".................................")
            st.text("HUMIDITY")
            st.text(".................................")
            st.write("Maximum = ", round(data[data['label']==crops_menu]["humidity"].max(),2))
            st.write("Minimum = ", round(data[data['label']==crops_menu]["humidity"].min(),2))
            st.write("Average = ", round(data[data['label']==crops_menu]["humidity"].mean(),2))
        with col5:    
            st.text(".................................")
            st.text("ph")
            st.text(".................................")
            st.write("Maximum = ", round(data[data['label']==crops_menu]["ph"].max(),2))
            st.write("Minimum = ", round(data[data['label']==crops_menu]["ph"].min(),2))
            st.write("Average = ", round(data[data['label']==crops_menu]["ph"].mean(),2))
        
        with col6:  
            st.text(".................................")
            st.text("RAINFALL")
            st.text(".................................")
            st.write("Maximum = ", round(data[data['label']==crops_menu]["rainfall"].max(),2))
            st.write("Minimum = ", round(data[data['label']==crops_menu]["rainfall"].min(),2))
            st.write("Average = ", round(data[data['label']==crops_menu]["rainfall"].mean(),2))  
with graph:
    allconditions=list(data.columns[:-1])
    condition_menu = st.selectbox('Select Condition to compare with other crops',(allconditions))
    pivot_table=data.groupby("label")[allconditions].mean().reset_index()
    if st.checkbox("Show graph"):
        fig=px.bar(pivot_table,x='label', y=condition_menu,width=600, height=400)
        fig.add_hline(y=data[condition_menu].mean(),line_color="red",annotation_text="Average", annotation_position="top right",fillcolor="red", opacity=0.25, line_width=5)
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),paper_bgcolor="LightSteelBlue")
        st.plotly_chart(fig.show())
        #print(pivot_table[[condition_menu]])
        #print("Overall Average =",round(data[condition_menu].mean(),2))        
    
    
with input:
    st.header('Predicting the Most Suitable crop based on selected conditions')
    Nitrogen=st.slider('WHAT IS THE AVERAGE SOIL NITROGEN',min_value=0,max_value=240)
    Phosphorus=st.slider('WHAT IS THE AVERAGE SOIL PHOSPHORUS',min_value=0,max_value=240)
    Potassium=st.slider('WHAT IS THE AVERAGE SOIL POTASSIUM',min_value=0,max_value=400)
    temperature=st.slider('WHAT IS THE AVERAGE TEMPERATURE',min_value=0,max_value=70)
    humidity=st.slider('WHAT IS THE AVERAGE SOIL HUMIDITY',min_value=0,max_value=200)
    ph=st.slider('WHAT IS THE AVERAGE SOIL PH',min_value=0,max_value=7)
    rainfall=st.slider('WHAT IS THE AVERAGE SOIL RAINFALL',min_value=0,max_value=600)
    



with model_training:
    st.header('PREDICTION')
    #st.text('Based on the selected crop conditions, the most suitable crop is:')

    #classifying crops into Summer crops, Winter crops and Rainy crops
    summer_crops=list(data[(data["temperature"]>30) & (data["humidity"]>50)]["label"].unique())
    winter_crops=list(data[(data["temperature"]<20) & (data["humidity"]>30)]["label"].unique())
    rainy_crops=list(data[(data["temperature"]>30) & (data["humidity"]>50)]["label"].unique())

    
    #lets duplicate the column label and name it crop so that we can then change crop into value labels
    #while maintaining label
    data["crop"]=data["label"]

    #Label Encoding
    #we will use cat.codes to label crop.
    #cat.codes works on categorical variables so we have to confirm the data type of label
    
    #checking the data type of 'label'
    #data["crop"].dtype #Data type is an object
    
    #lets change it from object to category
    #data["label"].astype('category').dtype
    
    #lets confirm the data type
    data["crop"]=data["crop"].astype('category')
    
    #As you had already observed that “crop” column datatype is an object type which is by default 
    #hence, need to convert “crop” to a category type with the help of pandas
    data["crop"]=data["crop"].cat.codes
    
    #Classifying the crops according to summer crops, winter crops and rainy crops.
    # the over twenty crops are too many for analysis
    def croptypes(x):
        if x in summer_crops:
            return "summer_crops"
        elif x in winter_crops:
            return "Winter crop"
        else:
            return "Rainy crop"
        
    data["crop_types"]=data["label"].apply(croptypes)
    
    #Support Vector Machine (SVM) Procedure
    # Defining Dependent (y) and Independent Variables (X)
    X=data.drop(columns=["label","crop","crop_types"]) 
    y=data["crop_types"]
    
    # Import module for splitting the data into train and test
    from sklearn.model_selection import train_test_split
    
    # Splitting the data in 80% train and 20% test
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2)
    
    # Import SVC module
    from sklearn.svm import SVC
    model_SVC=SVC()
    
    # Fitting the model to the training data
    model_SVC.fit(X_train, y_train)
    
    # Evaluating the performance of the model using score.
    model_SVC_performance="{:.2%}".format(model_SVC.score(X_test,y_test))
    
    prediction=list(model_SVC.predict([[Nitrogen,Phosphorus,Potassium,temperature,humidity,ph,rainfall]]))
    
    st.write("The most suitable crop given the conditions you have specified is ",prediction)
    if prediction == "summer_crops":
        st.write("Example: {}".format(summer_crops))
    elif prediction == "winter_crops":
        st.write("Example: {}".format(winter_crops))
    else:
        st.write("Example: {}".format(rainy_crops))

    st.write("PREDICTION ACCURACY=",model_SVC_performance)
    
