import pandas as pd
import numpy as np
import  nltk #natural language tool kit
import re #regular expression -removing the special characters
from  nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import models
from tensorflow.keras import layers

from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model

import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.

if os.environ.get('RUNTIME_ENV_LOCATION_TYPE') == 'external':
    endpoint_a4b346bcdab7475a847975ee9ac4f09c = 'https://s3.eu.cloud-object-storage.appdomain.cloud'
else:
    endpoint_a4b346bcdab7475a847975ee9ac4f09c = 'https://s3.private.eu.cloud-object-storage.appdomain.cloud'

client_a4b346bcdab7475a847975ee9ac4f09c = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='PZoBc-2q5JdiL5HJxx7eK0BMWF-wqmss_cr2kXAC0OXt',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url=endpoint_a4b346bcdab7475a847975ee9ac4f09c)

body = client_a4b346bcdab7475a847975ee9ac4f09c.get_object(Bucket='amazonkindle-donotdelete-pr-g9plr1qo7td0ze',Key='kindle_reviews update.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_1 = pd.read_csv(body)
df_data_1.head()

nltk.download("stopwords")
print(list(df_data_1.columns))


df=[]
for i in range(0,1000):
    review=df_data_1['reviewText'][i]
    #a)remove un neccessary .,
    review=re.sub('[^a-zA-Z]',' ',review)
    #b) lower case the text
    review=review.lower()
    #c)split the text
    review=review.split()
    #4.stemming
    #5. remove stop words
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #6.join the splitted data
    review=' '.join(review)
    df.append(review)

cv=CountVectorizer(max_features=6000)

x=cv.fit_transform(df).toarray()
print("Outputting Devi")
print(x)

y=df_data_1.iloc[0:1000:,3:4].values
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model = tensorflow.keras.Sequential()
model.add(layers.Dense(input_dim=9, units=1000, kernel_initializer="random_uniform",activation="relu")) # input layer
model.add(layers.Dense(units=9, kernel_initializer="random_uniform", activation="relu")) # 1st hidden layer
model.add(layers.Dense(units= 1, kernel_initializer="random_uniform", activation="sigmoid")) # output layer

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])

model.fit(x_train,y_train,batch_size=128,epochs=10)
a="amazo.h5"
pwd
tensorflow.keras.models.save_model(model,a)
!tar -zcvf Amazon_kindlereview.tgz amazo.h5
ls -1

!pip install watson-machine-learning-client --upgrade
!pip install --upgrade "ibm-watson>=5.2.3"

from ibm_watson_machine_learning import APIClient
wml_credentials = { "url": "https://eu-gb.ml.cloud.ibm.com",
                  "apikey": "i5pQI34nS5IUBRknr4pGgnsSe0N3D5SZkVdmGsMTehn4"
                  }
client = APIClient(wml_credentials)
def guide_from_space_name(client,space_name):
    space = client.spaces.get_details()
    return(next(item for item in space['resources']if item['entity']["name"]== space_name)['metadata']['id'])

space_uid = guide_from_space_name(client,'amazon')
print("Space UID = " + space_uid)

client.set.default_space(space_uid)
client.software_specifications.list()
software_spec_uid = client.software_specifications.get_uid_by_name("tensorflow_1.15-py3.6")

software_spec_uid

!pip install watson-machine-learning-client
model_details = client.repository.store_model(model='Amazon_kindlereview.tgz',
                                              meta_props= {
                                                  client.repository.ModelMetaNames.NAME:"NLP",
                                                  client.repository.ModelMetaNames.TYPE:"keras_2.2.4",
                                                  client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:software_spec_uid
                                              }) 
model_id = client.repository.get_model_uid(model_details)
model_id
ypred=model.predict(x_test)
ypred

model = load_model('amazo.h5')
#save bag of word model
import joblib
joblib.dump(cv.vocabulary_,"amazo.save")
loaded=CountVectorizer(decode_error='replace',vocabulary=joblib.load('amazo.save'))

d="Writing was good"
d=d.split('delimiter')
result=model.predict(loaded.transform(d))
print(result)
prediction=result<0.5
#print(prediction)
if prediction[0] == False:
    print("Positive review")
elif prediction[0] == True:
    print("Negative review")

