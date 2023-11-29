# Install required libraries
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# Read the two files
df1 = pd.read_csv('hfm_10cols.csv')
df2 = pd.read_csv('hfm_14cols.csv')
df3 = df2.drop(columns = ['Unnamed: 14','Unnamed: 15','label','time'])
df = pd.concat([df1,df3], axis=1)
df = df[['time','failure_label','sensor_1','sensor_2','sensor_3','sensor_4','sensor_5','sensor_6','sensor_7','sensor_8','sensor_9','sensor_10','sensor_11','sensor_12','sensor_13','sensor_14','sensor_15','sensor_16','sensor_17','sensor_18','sensor_19','sensor_20']]
df['time'] = pd.to_datetime(df['time'])



st.header(" 🛠️Machine Fault Predictor")



st.markdown(
    """
    <style>
 
    header {visibility: hidden;}
    .github-corner {display: none;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

def selected_chart():

    selected_chart = st.selectbox("👇 Key Sensors", ["Select a Sensor",
        "Sensor 1 vs. Time",
        "Sensor 2 vs. Time",
        "Sensor 3 vs. Time",
        "Sensor 4 vs. Time",
        "Sensor 5 vs. Time",
        "Sensor 6 vs. Time",
        "Sensor 7 vs. Time",
        "Sensor 8 vs. Time",
        "Sensor 9 vs. Time",
        "Sensor 10 vs. Time",
        "Sensor 11 vs. Time",
        "Sensor 12 vs. Time",
        "Sensor 13 vs. Time",
        "Sensor 14 vs. Time",
        "Sensor 15 vs. Time",
        "Sensor 16 vs. Time",
        "Sensor 17 vs. Time",
        "Sensor 18 vs. Time",
        "Sensor 19 vs. Time",
        "Sensor 20 vs. Time"
        
    ])

    if selected_chart != "Select a Sensor" and st.button("Generate Chart"):

        if selected_chart == "Sensor 1 vs. Time":
            for col in df.columns[2:3]:
                plt.rcParams['figure.figsize'] = [10,1]
                plt.scatter(df['time'],df[col])
                plt.xlabel('time')
                plt.ylabel(col)
                plt.show()
                st.pyplot(plt.gcf())

        if selected_chart == "Sensor 2 vs. Time":
            for col in df.columns[3:4]:
                plt.rcParams['figure.figsize'] = [10,1]
                plt.scatter(df['time'],df[col])
                plt.xlabel('time')
                plt.ylabel(col)
                plt.show()
                st.pyplot(plt.gcf())
        
        if selected_chart == "Sensor 3 vs. Time":
            for col in df.columns[4:5]:
                plt.rcParams['figure.figsize'] = [10,1]
                plt.scatter(df['time'],df[col])
                plt.xlabel('time')
                plt.ylabel(col)
                plt.show()
                st.pyplot(plt.gcf())

        if selected_chart == "Sensor 4 vs. Time":
            for col in df.columns[5:6]:
                plt.rcParams['figure.figsize'] = [10,1]
                plt.scatter(df['time'],df[col])
                plt.xlabel('time')
                plt.ylabel(col)
                plt.show()
                st.pyplot(plt.gcf())

        if selected_chart == "Sensor 5 vs. Time":
            for col in df.columns[6:7]:
                plt.rcParams['figure.figsize'] = [10,1]
                plt.scatter(df['time'],df[col])
                plt.xlabel('time')
                plt.ylabel(col)
                plt.show()
                st.pyplot(plt.gcf())

        if selected_chart == "Sensor 6 vs. Time":
            for col in df.columns[7:8]:
                plt.rcParams['figure.figsize'] = [10,1]
                plt.scatter(df['time'],df[col])
                plt.xlabel('time')
                plt.ylabel(col)
                plt.show()
                st.pyplot(plt.gcf())

        if selected_chart == "Sensor 7 vs. Time":
            for col in df.columns[8:9]:
                plt.rcParams['figure.figsize'] = [10,1]
                plt.scatter(df['time'],df[col])
                plt.xlabel('time')
                plt.ylabel(col)
                plt.show()
                st.pyplot(plt.gcf())

        if selected_chart == "Sensor 8 vs. Time":
            for col in df.columns[9:10]:
                plt.rcParams['figure.figsize'] = [10,1]
                plt.scatter(df['time'],df[col])
                plt.xlabel('time')
                plt.ylabel(col)
                plt.show()
                st.pyplot(plt.gcf())

        if selected_chart == "Sensor 9 vs. Time":
            for col in df.columns[10:11]:
                plt.rcParams['figure.figsize'] = [10,1]
                plt.scatter(df['time'],df[col])
                plt.xlabel('time')
                plt.ylabel(col)
                plt.show()
                st.pyplot(plt.gcf())   

        if selected_chart == "Sensor 10 vs. Time":
            for col in df.columns[11:12]:
                plt.rcParams['figure.figsize'] = [10,1]
                plt.scatter(df['time'],df[col])
                plt.xlabel('time')
                plt.ylabel(col)
                plt.show()
                st.pyplot(plt.gcf())

        if selected_chart == "Sensor 11 vs. Time":
            for col in df.columns[12:13]:
                plt.rcParams['figure.figsize'] = [10,1]
                plt.scatter(df['time'],df[col])
                plt.xlabel('time')
                plt.ylabel(col)
                plt.show()
                st.pyplot(plt.gcf())

        if selected_chart == "Sensor 12 vs. Time":
            for col in df.columns[13:14]:
                plt.rcParams['figure.figsize'] = [10,1]
                plt.scatter(df['time'],df[col])
                plt.xlabel('time')
                plt.ylabel(col)
                plt.show()
                st.pyplot(plt.gcf())
        
        if selected_chart == "Sensor 13 vs. Time":
            for col in df.columns[14:15]:
                plt.rcParams['figure.figsize'] = [10,1]
                plt.scatter(df['time'],df[col])
                plt.xlabel('time')
                plt.ylabel(col)
                plt.show()
                st.pyplot(plt.gcf())

        if selected_chart == "Sensor 14 vs. Time":
            for col in df.columns[15:16]:
                plt.rcParams['figure.figsize'] = [10,1]
                plt.scatter(df['time'],df[col])
                plt.xlabel('time')
                plt.ylabel(col)
                plt.show()
                st.pyplot(plt.gcf())

        if selected_chart == "Sensor 15 vs. Time":
            for col in df.columns[16:17]:
                plt.rcParams['figure.figsize'] = [10,1]
                plt.scatter(df['time'],df[col])
                plt.xlabel('time')
                plt.ylabel(col)
                plt.show()
                st.pyplot(plt.gcf())

        if selected_chart == "Sensor 16 vs. Time":
            for col in df.columns[17:18]:
                plt.rcParams['figure.figsize'] = [10,1]
                plt.scatter(df['time'],df[col])
                plt.xlabel('time')
                plt.ylabel(col)
                plt.show()
                st.pyplot(plt.gcf())

        if selected_chart == "Sensor 17 vs. Time":
            for col in df.columns[18:19]:
                plt.rcParams['figure.figsize'] = [10,1]
                plt.scatter(df['time'],df[col])
                plt.xlabel('time')
                plt.ylabel(col)
                plt.show()
                st.pyplot(plt.gcf())

        if selected_chart == "Sensor 18 vs. Time":
            for col in df.columns[19:20]:
                plt.rcParams['figure.figsize'] = [10,1]
                plt.scatter(df['time'],df[col])
                plt.xlabel('time')
                plt.ylabel(col)
                plt.show()
                st.pyplot(plt.gcf())

        if selected_chart == "Sensor 19 vs. Time":
            for col in df.columns[20:21]:
                plt.rcParams['figure.figsize'] = [10,1]
                plt.scatter(df['time'],df[col])
                plt.xlabel('time')
                plt.ylabel(col)
                plt.show()
                st.pyplot(plt.gcf())   

        if selected_chart == "Sensor 20 vs. Time":
            for col in df.columns[21:22]:
                plt.rcParams['figure.figsize'] = [10,1]
                plt.scatter(df['time'],df[col])
                plt.xlabel('time')
                plt.ylabel(col)
                plt.show()
                st.pyplot(plt.gcf())                                              

selected_chart()



def selected_model():

    selected_model = st.selectbox(" 👇 Machine Learning Models",
                                                                  
        ["Select a Model",
        "PCA",
        "AUTOENCODER",
        "K MEANS"])


    if selected_model != 'Select a Model' and st.button("Generate Output") :
        with st.spinner("Generating output..."):

            if selected_model == "PCA":
                   # Read the two files
                df1 = pd.read_csv('hfm_10cols.csv')
                df2 = pd.read_csv('hfm_14cols.csv')
                df3 = df2.drop(columns = ['Unnamed: 14','Unnamed: 15','label','time'])
                df = pd.concat([df1,df3], axis=1)
                df = df[['time','failure_label','sensor_1','sensor_2','sensor_3','sensor_4','sensor_5','sensor_6','sensor_7','sensor_8','sensor_9','sensor_10','sensor_11','sensor_12','sensor_13','sensor_14','sensor_15','sensor_16','sensor_17','sensor_18','sensor_19','sensor_20']]
                df['time'] = pd.to_datetime(df['time'])

                # Based on summary statistics dropping the sensor 4,5,9,15,17 and 18
                df = df.drop(columns = ['sensor_4','sensor_5','sensor_9','sensor_15','sensor_17','sensor_18'])
                df_normal = df[df['failure_label']==0]
                df_failure = df[df['failure_label']==1]
                # Train and test data
                X_train = df_normal.iloc[:,2:]
                X_fault = df_failure.iloc[:,2:]
                # Scaling the data
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                X_train_scaled = sc.fit_transform(X_train)
                X_test_scaled = sc.transform(X_fault)

                pca_redunction = PCA(n_components=10) # 90% variance in the data is captured, n_components=0.9
                X_train_pca = pca_redunction.fit_transform(X_train_scaled)
                X_test_pca = pca_redunction.transform(X_test_scaled)

                    # Increase the figure size
                plt.figure(figsize=(10, 6))
                    # Scatter plot for training data
                plt.scatter(X_train_pca, X_train_pca, c='blue', label='Train', edgecolor='black',alpha=0.5, s=50)

                    # Scatter plot for test data
                plt.scatter(X_test_pca, X_test_pca, c='red', label='Test', edgecolor='black',alpha=0.5, s=50)

                plt.xlabel('Dimension 1')
                plt.ylabel('Dimension 2')
                plt.legend()
                plt.title('Scatter Plot of PCA-Reduced Data')
                plt.show()
                st.pyplot(plt.gcf()) 

                def recon_loss(pca,X):
        #'pca' is the alredy fitted model on the "Fault-Free" data
                    X_pca = pca.transform(X)
                    X_recon = pca.inverse_transform(X_pca)
                    reconstruction_loss = np.mean((X-X_recon)**2,axis=1)
                    return reconstruction_loss
                
                st.markdown("Histogram for the reconstruction loss of the Fault Free data")
                Training_reconstruction_loss = recon_loss(pca=pca_redunction,X=X_train_scaled)
                plt.figure(figsize=(8, 4))
                plt.hist(Training_reconstruction_loss, bins=100,label='Train',alpha=1)
                # plt.xlim([0,80])
                plt.legend()
                plt.show()
                st.pyplot(plt.gcf()) 

                st.markdown("Histogram for the reconstruction loss of the Faulty data")
                Validation_reconstruction_loss = recon_loss(pca=pca_redunction,X=X_test_scaled)
                plt.figure(figsize=(8, 4))
                plt.hist(Validation_reconstruction_loss, bins=100,label='Val',alpha=1)
                # plt.xlim([0,80])
                plt.legend()
                plt.show()
                st.pyplot(plt.gcf()) 

                
                threshold = round(max(Training_reconstruction_loss),2)
                EntireData = df.iloc[:,2:]
                EntireData_scaled = sc.transform(EntireData)
                EntireData_reconstruction_loss = recon_loss(pca=pca_redunction,X=EntireData_scaled)
                plt.rcParams['figure.figsize'] = [8,3]
                plt.plot(EntireData_reconstruction_loss,label=f'recon_loss')
                plt.axhline(threshold,c='r',label='threshold')    #threshold value

                plt.title(f'Fault Detection using PCA')
                plt.legend()
                plt.ylim(0, 1.5)
                plt.show()
                st.pyplot(plt.gcf())


                # Counting number of samples above threshold
                mask = np.array(EntireData_reconstruction_loss) >= threshold
                count_true = np.sum(mask)
                indices = np.where(mask)
                selected_rows = df.iloc[indices]
                samples = selected_rows.shape[0]
                st.write(f"Number of samples above threshold = <span style='color: red;'>{samples}</span>", unsafe_allow_html=True)

                

    # Counting number of faults detected by model
                faults= (selected_rows.shape[0]-(sum(selected_rows['failure_label'] == 0)))
                st.write(f"Number of faults detected by model = <span style='color: red;'>{faults}</span>", unsafe_allow_html=True)

                Accuarcy = (selected_rows.shape[0]-(sum(selected_rows['failure_label'] == 0))) / selected_rows.shape[0]
                st.write(f"Accuarcy = <span style='color: red;'>{Accuarcy}</span>", unsafe_allow_html=True)


            
            if selected_model == "AUTOENCODER":


                # Read the two files
                df1 = pd.read_csv('hfm_10cols.csv')
                df2 = pd.read_csv('hfm_14cols.csv')
                df3 = df2.drop(columns = ['Unnamed: 14','Unnamed: 15','label','time'])
                df = pd.concat([df1,df3], axis=1)
                df = df[['time','failure_label','sensor_1','sensor_2','sensor_3','sensor_4','sensor_5','sensor_6','sensor_7','sensor_8','sensor_9','sensor_10','sensor_11','sensor_12','sensor_13','sensor_14','sensor_15','sensor_16','sensor_17','sensor_18','sensor_19','sensor_20']]
                df['time'] = pd.to_datetime(df['time'])

                # Based on summary statistics dropping the sensor 4,5,9,15,17 and 18
                df = df.drop(columns = ['sensor_4','sensor_5','sensor_9','sensor_15','sensor_17','sensor_18'])
                df_normal = df[df['failure_label']==0]
                df_failure = df[df['failure_label']==1]
                # Train and test data
                X_train = df_normal.iloc[:,2:]
                X_fault = df_failure.iloc[:,2:]
                # Scaling the data
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                X_train_scaled = sc.fit_transform(X_train)
                X_test_scaled = sc.transform(X_fault)


                input_layer = Input(shape=(X_train_scaled.shape[1],))

                # Define encoder layers
                encoded = Dense(32, activation='relu')(input_layer)

                # Latent space
                latent = Dense(16, activation='relu')(encoded)

                # Define decoder layers
                decoded = Dense(32, activation='relu')(latent)
                decoded = Dense(X_train_scaled.shape[1], activation='linear')(decoded)

                # Define autoencoder model
                autoencoder = Model(inputs=input_layer, outputs=decoded)

                # Compile autoencoder model
                autoencoder.compile(optimizer='adam', loss='mae')

                # Print model summary
                st.subheader("Autoencoder Model Summary")
                summary_str = []
                autoencoder.summary(print_fn=lambda x: summary_str.append(x))
                summary_text = '\n'.join(summary_str)

                # Display the model summary as plain text
                st.markdown(f"```plaintext\n{summary_text}\n```")

                # Define early stopping callback

                early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

                # Fit autoencoder model with early stopping callback
                history = autoencoder.fit(X_train_scaled, X_train_scaled,
                                epochs=100,
                                batch_size=32,
                                shuffle=True,
                                validation_data=(X_test_scaled, X_test_scaled),
                                callbacks=[early_stopping_callback])

                plt.rcParams['figure.figsize'] = [4, 2]
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title('Autoencoder Training and Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()
                st.pyplot(plt.gcf())

                # Function to obtain reconstruction loss
                def recon_loss(NN,X):
                    #NN is the trained neural network model
                    X_pred = NN.predict(X)
                    reconstruction_loss = np.mean((X-X_pred)**2,axis=1)
                    return reconstruction_loss

                st.markdown("Histogram for the reconstruction loss of the Fault Free data")
                Training_reconstruction_loss = recon_loss(NN=autoencoder,X=X_train_scaled)
                plt.figure(figsize=(8, 4))
                plt.hist(Training_reconstruction_loss, bins=100,label='Train',alpha=1)
                plt.legend()
                plt.show()
                st.pyplot(plt.gcf())


                st.markdown("Histogram for the reconstruction loss of the Faulty data")
                Validation_reconstruction_loss = recon_loss(NN=autoencoder,X=X_test_scaled)
                plt.figure(figsize=(8, 4))
                plt.hist(Validation_reconstruction_loss, bins=100,label='Val',alpha=1)
                # plt.xlim([0,80])
                plt.legend()
                plt.show()
                st.pyplot(plt.gcf())

                st.markdown("Fault detection using residuals for the entire data using threshold")
                threshold = round(max(Training_reconstruction_loss),2)
                EntireData = df.iloc[:,2:]
                EntireData_scaled = sc.transform(EntireData)
                EntireData_reconstruction_loss = recon_loss(NN=autoencoder,X=EntireData_scaled)


                plt.rcParams['figure.figsize'] = [8, 3]
                plt.plot(EntireData_reconstruction_loss,label=f'recon_loss')
                plt.axhline(threshold,c='r',label='threshold')    #threshold value
                plt.title(f'Fault Detection using AE')
                plt.legend()
                plt.ylim(0, 0.2)
                plt.show()
                st.pyplot(plt.gcf())

                # Counting number of samples above threshold
                mask = np.array(EntireData_reconstruction_loss) >= threshold
                count_true = np.sum(mask)
                indices = np.where(mask)
                selected_rows = df.iloc[indices]
                
                st.write(f"Number of samples above threshold = <span style='color: red;'>{selected_rows.shape[0]}</span>", unsafe_allow_html=True)


                # Counting number of faults detected by model
                NUM2 = (selected_rows.shape[0]-(sum(selected_rows['failure_label'] == 0)))
                st.write(f"Number of faults detected by model = <span style='color: red;'>{NUM2}</span>", unsafe_allow_html=True)


                #Calculationg accuracy
                Accuarcy = (selected_rows.shape[0]-(sum(selected_rows['failure_label'] == 0))) / selected_rows.shape[0]
                st.write(f"Accuarcy = <span style='color: red;'>{Accuarcy}</span>", unsafe_allow_html=True)

        if selected_model == "K MEANS":

            # Read the two files
            df1 = pd.read_csv('hfm_10cols.csv')
            df2 = pd.read_csv('hfm_14cols.csv')
            df3 = df2.drop(columns = ['Unnamed: 14','Unnamed: 15','label','time'])
            df = pd.concat([df1,df3], axis=1)
            df = df[['time','failure_label','sensor_1','sensor_2','sensor_3','sensor_4','sensor_5','sensor_6','sensor_7','sensor_8','sensor_9','sensor_10','sensor_11','sensor_12','sensor_13','sensor_14','sensor_15','sensor_16','sensor_17','sensor_18','sensor_19','sensor_20']]
            df['time'] = pd.to_datetime(df['time'])
            df = df.drop(columns = ['sensor_4','sensor_5','sensor_9','sensor_15','sensor_17','sensor_18'])
            df_normal = df[df['failure_label']==0]
            df_failure = df[df['failure_label']==1]
            # Train and test data
            X_train = df_normal.iloc[:,2:]
            X_fault = df_failure.iloc[:,2:]
            # Scaling the data
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train_scaled = sc.fit_transform(X_train)
            X_test_scaled = sc.transform(X_fault)

            # Define the number of clusters
            k = 1

            # Fit k-means clustering model
            kmeans = KMeans(n_clusters=k).fit(X_train_scaled)

            # Function to calculate the Distance of Fault free data from centre
            def distanceFromCenter(cluster_center,X):
                p_dist=[]
                for i in range(len(X)):
                    dist= pairwise_distances(cluster_center, X[i].reshape(1, -1))
                    p_dist.append(dist)
                return np.array(p_dist).reshape(-1)

            st.markdown("Histogram for the reconstruction loss of the Fault Free data")
            Training_reconstruction_loss=distanceFromCenter(cluster_center=kmeans.cluster_centers_,X=X_train_scaled)

            plt.rcParams['figure.figsize'] = [4, 4]
            plt.hist(Training_reconstruction_loss, bins=100,label='FaultFree',alpha=1)
            plt.legend()
            plt.show()
            st.pyplot(plt.gcf())

            st.markdown("Histogram for the reconstruction loss of the Faulty data")
            faulty_dist = distanceFromCenter(cluster_center=kmeans.cluster_centers_,X=X_test_scaled)
            plt.rcParams['figure.figsize'] = [4, 4]
            plt.hist(faulty_dist, bins=100,label='FaultFree',alpha=1)
            plt.legend()
            plt.show()
            st.pyplot(plt.gcf())

            # Fault detection using residuals for the entire data using threshold
            threshold = math.ceil(max(Training_reconstruction_loss))
            EntireData = df.iloc[:,2:]
            EntireData_scaled = sc.transform(EntireData)
            EntireData_reconstruction_loss = distanceFromCenter(cluster_center=kmeans.cluster_centers_,X=EntireData_scaled)

            plt.rcParams['figure.figsize'] = [8, 3]
            plt.plot(EntireData_reconstruction_loss,label=f'recon_loss')
            plt.axhline(threshold,c='r',label='threshold')    #threshold value

            plt.title(f'Fault Detection using K-Means')
            plt.ylim(0, 25)
            plt.legend()
            plt.show()
            st.pyplot(plt.gcf())

            # Counting number of samples above threshold
            mask = np.array(EntireData_reconstruction_loss) >= threshold
            count_true = np.sum(mask)
            indices = np.where(mask)
            selected_rows = df.iloc[indices]       
            st.write(f"Number of samples above threshold = <span style='color: red;'>{selected_rows.shape[0]}</span>", unsafe_allow_html=True)


            # Counting number of faults detected by model
            num2 = (selected_rows.shape[0]-(sum(selected_rows['failure_label'] == 0)))
            st.write(f"Number of faults detected by model = <span style='color: red;'>{num2}</span>", unsafe_allow_html=True)

            Accuarcy = (selected_rows.shape[0]-(sum(selected_rows['failure_label'] == 0))) / selected_rows.shape[0]
            st.write(f"Accuarcy = <span style='color: red;'>{Accuarcy}</span>", unsafe_allow_html=True)


selected_model()
