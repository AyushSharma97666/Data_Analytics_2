# Data_Analytics_2


# Lung Cancer Risk Prediction

Lung Cancer is growing now days. we are analyzing  lung cancer data, we are useing kaggle [Lung Cancer](https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer) dataset. In it, predicting risk of lung cancer to a person. we are useing EDA.
Exploratory Data Analysis (EDA) is an approach to analyze the data using visual techniques. It is used to discover trends, patterns, or to check assumptions with the help of statistical summary and graphical representations.


**Acknowledgments**       
When we use this dataset in our research, we credit the authors as :
- License : CC BY 4.0.
- Hong, Z.Q. and Yang, J.Y. "Optimal Discriminant Plane for a Small Number of Samples and Design Method of Classifier on the Plane", Pattern Recognition, Vol. 24, No. 4, pp. 317-324, 1991 and it is published t to reuse in [google research dataset](https://datasetsearch.research.google.com/search?src=0&query=Lung%20Cancer&docid=L2cvMTFqY2p2X3B2Yg%3D%3D)

## ASK  
**Main Objects :- predicting risk of lung cancer**

## Prepare 
**Dataset :- Lung Cancer (kaggle)** 
We are useing kaggle dataset for EDA. Data is provide by kaggle, no need to check it's vaildation. it is availble [here](https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer).   
Dataset details :- 
Total no. of attributes:16 & No .of instances:284 
Attribute information:           
|1.Gender | 2.Age | 3.Smoking |4.Yellow fingers | 5.Anxiety |6.Peer_pressure |7.Chronic Disease | 8.Fatigue | 9.Allergy | 10.Wheezing  |11.Alcohol | 12.Coughing | 13.Shortness of Breath | 14.Swallowing Difficulty | 15.Chest pain | 16.Lung Cancer |
|------ | ------ | ------|------ | ------ |------|------ | ------| ------ | ------ |------ | ------ | ------| ------ | ------ | ------ |
|M(male), F(female) | Age of the patient |  YES=2 , NO=1| YES=2 , NO=1 |  YES=2 , NO=1 | YES=2 , NO=1| YES=2 , NO=1 |  YES=2 , NO=1|  YES=2 , NO=1 |  YES=2 , NO=1 | YES=2 , NO=1 |  YES=2 , NO=1 |  YES=2 , NO=1|  YES=2 , NO=1 |  YES=2 , NO=1 | YES , NO. |


## Process
For cleaning data and maintaning data, for it we will use python. 
Loading data into pandas dataframe :    


```   
import pandas as pd    
#importing data into pandas dataframe     
df=pd.read_csv("survey lung cancer.csv", index_col=0)   
df.inf0()
```
  
![image](https://user-images.githubusercontent.com/110818513/191474115-d48368a1-8ea0-4c11-8c43-e30e605e25b2.png)

From above we know that there is no null value.

## Analyze
Starting analysis, useing python dataframe i.e., pandas mostly 
we have 309 patients details, but not everyone have cancer. we are separating data of those patients have the disease.

```    
# grouping data for finding how many people have cancer 
df_1=df.groupby(["LUNG_CANCER"])['LUNG_CANCER'].count()
df_1.plot.pie(autopct="%.2f")  
```   
![image](https://user-images.githubusercontent.com/110818513/191475505-02a22220-fc3a-49d7-bd66-8bb060d85753.png)

From above, 83% person in data have cancer. it is good for to predicating risk. Now we analyze data who have disease, to get for details.

We are grouping data by Gender, to look more into data.

```
#Filtering data, person have cancer 
df_2=df[df['LUNG_CANCER']== 'YES']

# grouping data by gender  
df_3=df_2.groupby(['GENDER'])['GENDER'].count()
df_3.plot.pie(autopct="%.2f")

```
![image](https://user-images.githubusercontent.com/110818513/191477402-49992782-b48c-4d3c-a479-292053ce9bf3.png)

for above, there is no any big difference between male and women those have cancer. Now we analyze data according by age 


```  
# list of index
list_index=df_2.index.tolist()

# Disturbing Age of the patient into age range 
age_range= []
for i in list_index :
    if  100 >=df_2['AGE'][i] >90:
        age_range.append("91-100")
    elif  90 >=df_2['AGE'][i] >80:
        age_range.append("81-90")
    elif  80 >=df_2['AGE'][i] >70:
        age_range.append("71-80")
    elif  70 >=df_2['AGE'][i] >60:
        age_range.append("61-70")
    elif 60 >= df_2['AGE'][i] > 50 :
         age_range.append("51-60")
    elif 50 >= df_2['AGE'][i] > 40 :
         age_range.append("41-50")
    elif 40 >= df_2['AGE'][i] > 30 :
         age_range.append("31-40")
    elif df_2['AGE'][i]<= 30  :
        age_range.append("below 30 ")
    else :
        age_range.append("above 100")
        
# Now adding new list into adding age_range list into pandas dataframe
df_2['Age_range']=age_range    

# now group data from age_range 
df_3=df_2.groupby(['Age_range'])['Age_range'].count()
df_3.plot.pie(autopct="%.2f")

```

![image](https://user-images.githubusercontent.com/110818513/191481017-0f72592e-099c-46e0-9001-03026e47ca10.png)

Above shows, patient those have age above 50 are have more risk of disease. It shows age above 50 have 94%.

Adding data of Attribute  into one column and grouping these data analyzing more it.

```
# filtering data, patient with cancer and age above 50 
df_4=df_2[df_2['AGE']>= 50]

# creating list of index
list_index_1=df_4.index.tolist()
list_index_1

# creating new columns, adding all attribute into one column 
#new column list 
final_list_disease=[]

#adding attribute into one list 
for i in list_index_1:
    disease=''
    
    if df_4['SMOKING'][i] == 2 :
        disease = disease + 'SMOKING '
    if df_4['YELLOW_FINGERS'][i] == 2 :
        disease = disease + 'YELLOW_FINGERS '
    if df_4['ANXIETY'][i] == 2 :
        disease = disease + 'ANXIETY '
    if df_4['PEER_PRESSURE'][i] == 2 :
        disease = disease + 'PEER_PRESSURE '
    if df_4['CHRONIC DISEASE'][i] == 2 :
        disease = disease + 'CHRONIC DISEASE '
    if df_4['WHEEZING'][i] == 2 :
        disease = disease + 'WHEEZING '
    if df_4['ALCOHOL CONSUMING'][i] == 2 :
        disease = disease + 'ALCOHOL CONSUMING '
    if df_4['COUGHING'][i] == 2 :
        disease = disease + 'COUGHING '
    if df_4['SHORTNESS OF BREATH'][i] == 2 :
        disease = disease + 'SHORTNESS OF BREATH '
    if df_4['SWALLOWING DIFFICULTY'][i] == 2 :
        disease = disease + 'SWALLOWING DIFFICULTY '
    if df_4['CHEST PAIN'][i] == 2 :
        disease = disease + 'CHEST PAIN '  
    if df_4['SMOKING'][i]== df_4['YELLOW_FINGERS'][i]== df_4['ANXIETY'][i]== df_4['PEER_PRESSURE'][i]== df_4['CHRONIC DISEASE'][i]== df_4['WHEEZING'][i]==    df_4['ALCOHOL CONSUMING'][i]== df_4['COUGHING'][i]== df_4['COUGHING'][i]== df_4['SHORTNESS OF BREATH'][i]== df_4['SWALLOWING DIFFICULTY'][i]== df_4['CHEST PAIN'][i] == 1 :
        disease = disease + ' '
      
    final_list_disease.append(disease)

# adding new list all attribute into one column in pandas dataframe
df_4['Grouped disease']=final_list_disease

# grouping data according new column 
df_5=df_4.groupby(['Grouped disease'])['Grouped disease'].count().reset_index(name='count')

# filtering data those have count is greater than 5 
df_6= df_5[df_5['count'] >= 5]

#sorting data according 
df_6.sort_values(by=['count'],  ascending=False).reset_index()

```

![image](https://user-images.githubusercontent.com/110818513/191484597-0ab26848-8fbe-41e7-9a7d-ae77c7f9a198.png)

Above shows that, person have SMOKING ,WHEEZING, ALCOHOL CONSUMING COUGHING, SHORTNESS OF BREATH, CHEST PAIN are on more risk of getting lung cancer.

```
# plotting  horizontal bar 
ax= df_6.plot.barh(x='Grouped disease', y='count')

```

![image](https://user-images.githubusercontent.com/110818513/191485689-6266253c-3830-432b-a8d4-e73bdd28946e.png)

## Share

After analyzing, who can predicate people can have or grow lung cancer due different attribute and risk of have lung cancer.So the people can take the appropriate decision based on their cancer risk status.
**Following are findings:-**
- Patient have attribute these SMOKING ,WHEEZING, ALCOHOL CONSUMING COUGHING, SHORTNESS OF BREATH, CHEST PAIN  are most. 
- There is small difference in gender those have Lung Cancer, so we can say that, it not gender affectes due to gender
- People have age above 50 are more vulnerable to the disease, data shows 94% patients have the disease and there is age above 50.

**Thank you very much for your time, I hope you enjoyed the reading**     
**I would be very happy to read your comments and feedback**
