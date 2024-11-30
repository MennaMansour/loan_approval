import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report ,confusion_matrix , ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from imblearn.combine import SMOTEENN
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import pickle


# configure page
st.set_page_config(layout='wide')
df = pd.read_csv('Loan approval prediction.csv', encoding="ISO-8859-1")
placeholder = st.empty()
df_copy=df.copy()
df_copy = df_copy.drop(columns=['id'])
########################################################

st.sidebar.title("Loan Approval")
option = st.sidebar.selectbox("Pick a choice:",['Home','EDA','ML'])

if option == 'Home':
    st.title("Hello,Loan approval")
    st.subheader('1- sample of data')
    # handling outliers
    df_copy = df_copy[(df_copy['person_age'] <= 65) & (df_copy['person_age'] >= 20)]
    df_copy = df_copy[(df_copy['person_income'] <= 200000)]
    df_copy = df_copy[(df_copy['person_emp_length'] <= 31)]
    df[df['loan_percent_income']>=0.55]
    df_copy = df_copy[(df_copy['loan_percent_income'] <= 0.55)]
    st.markdown(
    "<hr style='border:2px solid #4CAF50; margin:20px 0;'>", 
    unsafe_allow_html=True
    )
###########################################
    st.subheader("2- Shape of the dataset")
    st.write(df.shape)
    st.subheader("3-statistical information about dataset")
    st.text('numeric')
    st.write(df_copy.describe())
    st.text('categoric')
    st.write(df_copy.describe(include='object'))
    st.markdown(
    "<hr style='border:2px solid #4CAF50; margin:20px 0;'>", 
    unsafe_allow_html=True
    )
    st.subheader("4- check for missing value")
    st.write(df_copy.isna().sum())
    st.subheader("5- counts of unique values in each feature")
    st.markdown(
    "<hr style='border:2px solid #4CAF50; margin:20px 0;'>", 
    unsafe_allow_html=True
    )
######################################################################
    st.text('numeric')
    objcols = df.select_dtypes(include=['O', 'category']).columns
    for col in objcols:
       st.write('='*40)
       st.write(df[col].value_counts())
       st.write('='*40)
       st.markdown(
    "<hr style='border:2px solid #4CAF50; margin:20px 0;'>", 
    unsafe_allow_html=True
    )
       st.text('categoric')
    objcols = df.select_dtypes(include=["int64",'float64']).columns ### There is an imbalance
    for col in objcols:
        st.write(df[col].value_counts())
        st.write('='*40)

###############################################################
elif option == 'EDA':
    st.title("Hello,Loan approval")
    counts = df_copy['loan_status'].value_counts()
    st.subheader('1-Distribution of loan_status')
    fig = px.pie(counts, values=counts.values, names=counts.index,)
    st.plotly_chart(fig)
    st.text('observation:   data is impalance')
    st.markdown(
    "<hr style='border:2px solid #2B60DE; margin:20px 0;'>", 
    unsafe_allow_html=True
    )
#####################################################################################
    st.subheader('2- Box blot to show outliers')
    num_df=df.select_dtypes(include=['int64','float64']).columns
    selected_column = st.selectbox('Select a column :', num_df)
    fig = px.box(num_df, y=df[selected_column])
    fig.update_layout(yaxis_title=selected_column)
    st.plotly_chart(fig)
    st.markdown(
    "<hr style='border:2px solid #2B60DE; margin:20px 0;'>", 
    unsafe_allow_html=True
    )
########################################################################################
    st.subheader('3- Distribution of Some Features')
    features = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate','person_emp_length','cb_person_cred_hist_length']
    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(features, 1):
        plt.subplot(3, 3, i)
        sns.histplot(df[feature], bins=30, kde=True)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    st.pyplot(plt)
    st.markdown("""
    - **Point 1**:The majority of borrowers fall into the 20-30 age group, indicating younger to middle-aged individuals dominate the dataset.
    - **Point 2**: Loan amounts are skewed toward lower values, with most requests under $15,000.
    - **Point 3**: most of them have person income from 25000 to 100000.
    - **Point 3**: most of them have emp_length lower than 15, higther from(0 to 5).
    """)
    st.markdown(
    "<hr style='border:2px solid #2B60DE; margin:20px 0;'>", 
    unsafe_allow_html=True
    )
    #############################################################################
    st.subheader('4-relationship of Some Features  with  loan_status ')
    columns = ['cb_person_default_on_file', 'person_home_ownership', 'loan_grade', 'loan_intent']

    selected_column = st.selectbox('Select a column to group by:', columns)
    grouped_data = df_copy.groupby([df_copy[selected_column], 'loan_status']).size().reset_index(name='count')

    fig = px.bar(
        grouped_data,
        x=selected_column,
        y='count',
        color='loan_status',
        title=f'Number of loan_status and {selected_column}')

    fig.update_xaxes(title_text=selected_column)
    fig.update_yaxes(title_text='Count')
    fig.update_traces(
    text=grouped_data['count'],
    textposition='outside'
          )

    st.plotly_chart(fig)
    st.markdown("""
    - **loan_intent**:
       - EDUCATION have higher count when loan status =0
       - MEDICAL AND DEPTCONSOLDATION have higher count when loan status =1
    - **cb_person_default_on_file**:
       - Individuals with a history of defaults (cb_person_default_on_file = Y) have lower approval rates.
       - Borrowers with no defaults (cb_person_default_on_file = N) have higher approval rates.
    - **loan_grade**: - Higher grades (A, B) should have lower interest rates
       - Lower grades (E, F, G) are expected to have higher interest rates ,because higher  defaulting
       - D is higher aproved loan because have person_home_ownership  rent and mortage higher ration
       - Higher grades (A, B) should have lower default rates
    """)
    st.markdown(
    "<hr style='border:2px solid #2B60DE; margin:20px 0;'>", 
    unsafe_allow_html=True
    )

    ##################################################################

    st.subheader("5-Distribution of Loan Interest Rate Across Loan Grades")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_copy, x='loan_grade', y='loan_int_rate', palette='coolwarm', ax=ax)
    ax.set_title('Distribution of Loan Interest Rate Across Loan Grades')
    ax.set_xlabel('Loan Grade')
    ax.set_ylabel('Loan Interest Rate (%)')
    st.pyplot(fig)
    #################################################################
 
    loan_grade_default_rate = df_copy.groupby('loan_grade')['loan_status'].mean().reset_index()

    st.subheader("Loan Analysis: Default Rate and Count by Loan Grade")
    st.subheader("Loan Default Rate by Loan Grade")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=loan_grade_default_rate, x='loan_grade', y='loan_status', palette='coolwarm', ax=ax1)
    ax1.set_title("Loan Default Rate by Loan Grade")
    ax1.set_xlabel("Loan Grade")
    ax1.set_ylabel("Default Rate")
    st.pyplot(fig1)

    st.subheader("Default Count by Loan Grade")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.countplot(data=df_copy, x='loan_grade', hue='loan_status', palette='coolwarm', ax=ax2)
    ax2.set_title("Default Count by Loan Grade")
    ax2.set_xlabel("Loan Grade")
    ax2.set_ylabel("Count of Loans")
    ax2.legend(title="Loan Status", labels=["No Default", "Default"])
    st.pyplot(fig2)
    ###################################################
    st.subheader("Average Loan Amount by Loan Grade and Status")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='loan_grade', y='loan_amnt', hue='loan_status', data=df_copy, estimator=np.mean, ax=ax)
    ax.set_title('Average Loan Amount by Loan Grade and Status')
    ax.set_xlabel('Loan Grade')
    ax.set_ylabel('Average Loan Amount')
    ax.legend(title='Loan Status')
    st.pyplot(fig)
    st.markdown(
    "<hr style='border:2px solid #2B60DE; margin:20px 0;'>", 
    unsafe_allow_html=True
    )
    ######################################################
    st.subheader("5-relationship between feature")
    st.text("Choose Columns for Scatter Plot")
    columns = df_copy.columns  # Get column names from the DataFrame
    x_column = st.selectbox("Select X-axis Column", columns)
    y_column = st.selectbox("Select Y-axis Column", columns)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
    x=df_copy[x_column],
    y=df_copy[y_column],
    alpha=0.7,
    edgecolor=None,
    ax=ax
   )
    ax.set_title(f"Relationship Between {x_column} and {y_column}", fontsize=10)
    ax.set_xlabel(x_column, fontsize=12)
    ax.set_ylabel(y_column, fontsize=12)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    
    st.markdown(
    """
    - positive Correlation between person_age and requested cb_person_cred_hist_length.
    - positive Correlation between income and requested loan amount.
    - positive Correlation between Age and requested emplolyment length.
    """) 
    st.markdown(
    "<hr style='border:2px solid #2B60DE; margin:20px 0;'>", 
    unsafe_allow_html=True
    )
    ############################################################################33
    st.subheader("6-correlation  between feature")
    numeric_df = df_copy.select_dtypes(include=['number'])

    plt.figure(figsize=(20, 15))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    st.pyplot(plt)


    #########################################################  
elif option == "ML":
    df_copy['disposable_income'] = df_copy['person_income'] - (df_copy['loan_amnt'] * (1 + df_copy['loan_int_rate']))
    def employment_stability(emp_length):
        if emp_length < 5:
           return 'Unstable'
        elif 5 <= emp_length <= 10:
           return 'Relatively Stable'
        else:
          return 'Highly Stable'

    df_copy['employment_stability'] = df_copy['person_emp_length'].apply(employment_stability)

    def credit_history_stability(hist_length):
        if hist_length < 5:
           return 'Short'
        elif 5 <= hist_length <= 10:
           return 'Medium'
        else:
           return 'Long'

    df_copy['credit_history_stability'] = df_copy['cb_person_cred_hist_length'].apply(credit_history_stability)
    df_copy['income_credit_interaction'] = df_copy['person_income'] * df_copy['cb_person_cred_hist_length']
########################################################################################################################
    label_encoder = LabelEncoder()
    columns_to_encode = df_copy.select_dtypes(exclude=['number'])
    for column in columns_to_encode:
         df_copy[column] = label_encoder.fit_transform(df_copy[column])
    
    columns_to_scale = ['person_age', 'person_income',  
                    'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
                    'disposable_income',
                    'income_credit_interaction'
       ]

    scaler = StandardScaler()
    df_copy[columns_to_scale] = scaler.fit_transform(df_copy[columns_to_scale])

###############################################################################################
    x=df_copy[['person_age', 'person_income', 'person_home_ownership', 'loan_intent', 'loan_grade', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file',
       'disposable_income', 'employment_stability',
       'credit_history_stability', 'income_credit_interaction']]
    y=df_copy[['loan_status']] 
    
    smote = SMOTEENN(sampling_strategy=0.25,random_state=42)
    X_resampled, y_resampled = smote.fit_resample(x,y)    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)  

    
#############################################################################
      

    # Load pre-trained models
    models = {
        "Decision Tree": pickle.load(open("dtree.pkl", "rb")),
        "Random Forest": pickle.load(open("Rforest.pkl", "rb")),
        "KNN": pickle.load(open("knn.pkl", "rb")),
        "Logistic Regression": pickle.load(open("log.pkl", "rb")),
        "SVM": pickle.load(open("svm_model.pkl", "rb")),
    }

    # Initialize the scaler and label encoder
    scaler = StandardScaler()
    le = LabelEncoder()

    # Streamlit UI setup
    st.title("Hello,Loan Status Prediction")
    st.text("Predict loan approval status using different machine learning models.")

    # Collect user input for features
    person_age = st.number_input("Enter person's age", key="person_age")
    person_income = st.number_input("Enter person's income", key="person_income")
    person_home_ownership = st.selectbox(
        "Select homeownership status", ["RENT", "OWN", "MORTGAGE",'OTHER'], key="person_home_ownership"
    )
    loan_intent = st.selectbox(
        "Select loan intent", ["PERSONAL", "EDUCATION", "HOMEIMPROVEMENT",'MEDICAL','DEBTCONSOLIDATION','VENTURE'], key="loan_intent"
    )
    loan_grade = st.selectbox(
        "Select loan grade", ["A", "B", "C", "D", "E", "F", "G"], key="loan_grade"
    )
    loan_amnt = st.number_input("Enter loan amount", key="loan_amnt")
    loan_int_rate = st.number_input("Enter loan interest rate", key="loan_int_rate")
    loan_percent_income = st.number_input("Enter loan percentage of income", key="loan_percent_income")
    cb_person_default_on_file = st.selectbox(
        "Has the person defaulted on credit before?", ["Y", "N"], key="cb_person_default_on_file"
    )

    disposable_income = st.number_input("Enter disposable income", key="disposable_income")
    employment_stability = st.selectbox(
        "Enter employment stability", ["Unstable", "Relatively Stable", "Highly Stable"],
        key="employment_stability"
    )
    credit_history_stability = st.selectbox(
        "Enter credit history stability", ["Short", "Medium", "Long"], key="credit_history_stability"
    )
    income_credit_interaction = st.number_input("Enter income-credit interaction", key="income_credit_interaction")

    # Combine the features into a DataFrame
    input_data = pd.DataFrame([{
        'person_age': person_age,
        'person_income': person_income,
        'person_home_ownership': person_home_ownership,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_default_on_file': cb_person_default_on_file,
        'disposable_income': disposable_income,
        'employment_stability': employment_stability,
        'credit_history_stability': credit_history_stability,
        'income_credit_interaction': income_credit_interaction
    }])

    # Encode categorical features
    categorical_columns = [
        'person_home_ownership', 'loan_intent', 'loan_grade'
        , 'employment_stability', 'credit_history_stability','cb_person_default_on_file'
    ]
    for col in categorical_columns:
        input_data[col] = le.fit_transform(input_data[col])

    # Scale numerical features
    numerical_columns = [
        'person_age', 'person_income', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income',
        'disposable_income', 'income_credit_interaction'
    ]
    input_data[numerical_columns] = scaler.fit_transform(input_data[numerical_columns])

    # Dropdown for model selection
    model_choice = st.selectbox("Choose a model for prediction", list(models.keys()))

    # Add a button for prediction
    if st.button("Predict Loan Status"):
        # Get the selected model
        selected_model = models[model_choice]
        
        # Make the prediction
        prediction = selected_model.predict(input_data)[0]
        
        # Display the result
        if prediction == 1:
            st.success(f"Prediction with {model_choice}: Loan Approved")
        else:
            st.error(f"Prediction with {model_choice}: Loan Denied")

