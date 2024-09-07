import numpy as np
# ImportError: Numba needs NumPy 1.21 or less
import pandas as pd
import streamlit as st
# from pandas_profiling import ProfileReport
import pandas as pd
from ydata_profiling import ProfileReport
# from ydata_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import altair as alt
import plotly.express as px
import plotly.figure_factory as ff

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn  as sns

# from pydantic_settings import BaseSettings




def main():
    html_temp1 = """<div style="background-color:#6D7B8D;padding:10px">
                            		<h4 style="color:white;text-align:center;">Exploratory data Analysis Application</h4>
                            		</div>
                            		<div>
                            		</br>"""
    st.markdown(html_temp1, unsafe_allow_html=True)

    menu = ["Home", "EDA", "About","arab","india"]
    choice = st.sidebar.selectbox("Menu", menu, 2)
    # for hide menu
    hide_streamlit_style = """
                    <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    </style>
                    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.sidebar.markdown(
        """ Developed by Mohammad Juned Khan    
            Email : Mohammad.juned.z.khan@gmail.com  
            [LinkedIn] (https://www.linkedin.com/in/md-juned-khan)""")
    if choice == "Home":
        # color codes  ff1a75  6D7B8D
        html_temp2 = """<div style="background-color:#6D7B8D;padding:10px">
                                        		<h4 style="color:white;text-align:center;">This is the Exploratory data Analysis Application  created using Streamlit framework and pandas-profiling library.</h4>
                                        		</div>
                                        		<div>
                                        		</br>"""
        st.markdown(html_temp2, unsafe_allow_html=True)

    # elif choice == "EDA":
    #     html_temp3 = """
    #                     		<div style="background-color:#98AFC7;padding:10px">
    #                     		<h4 style="color:white;text-align:center;">Upload file Your file in csv formate and perform Exploratory Data Analysis</h4>
    #                     		<h5 style="color:white;text-align:center;">Make sure your columns have correct data types before uploading.</h5>
    #                     		</div>
    #                     		<br></br>"""
    #
    #     st.markdown(html_temp3, unsafe_allow_html=True)
    #     st.subheader("Perform Exploratory data Analysis with Pandas Profiling Library")
    #     data_file = st.file_uploader("Upload a csv file", type=["csv"])
        if st.button("Analyze"):
            if data_file is not None:
                # Pandas Profiling Report
                @st.cache
                def load_csv():
                    csv = pd.read_csv(data_file)
                    return csv

                df = load_csv()
                pr = ProfileReport(df, explorative=True)
                st.header('*User Input DataFrame*')
                st.write(df)
                st.write('---')
                st.header('*Exploratory Data Analysis Report Using Pandas Profiling*')
                st_profile_report(pr)

            else:
                st.success("Upload file")
        else:
            pass
        # st.write("Check similarity of Resume and Job Description")
    elif choice == "About":
        html_temp4 = """
                       		<div style="background-color:#98AFC7;padding:10px">
                       		<h4 style="color:white;text-align:center;">This Application is developed by Mohammad Juned Khan using Streamlit Framework. If you're on LinkedIn and want to connect, just click on the link in sidebar and shoot me a request. You can also mail your comments. </h4>
                       		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>

                       		</div>
                       		<br></br>
                       		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)
    elif choice == "arab":
        st.subheader("Perform Exploratory data Analysis with Pandas Profiling Library")
        data_file = st.file_uploader("Upload a csv file", type=["csv"])
        if st.button("Analyze"):
            if data_file is not None:
                # # Pandas Profiling Report
                # @st.cache_data
                # def load_csv():
                #     csv = pd.read_csv(data_file)
                #     return csv
                # df = load_csv
                df = pd.read_csv(data_file)
                st.dataframe(df)
                c = df.head()
                st.write(c)
                # import streamlit as st
                # import pandas as pd
                # import numpy as np
                # import seaborn as sns
                # import matplotlib.pyplot as plt
                # import altair as alt
                # import plotly.express as px
                # import plotly.figure_factory as ff
                #
                # st.subheader('1.UPLOAD DATA')
                # df = pd.read_csv('gdp.csv')
                # st.dataframe(df)

                # st.subheader('INTERACTIVE CHART')
                # country_name = df.columns.tolist()
                # country_choices = st.multiselect('choose country',country_name)
                # new_df=df[country_choices]
                # st.line_chart(new_df)

                data = df.describe()
                data

                st.subheader('2.Distribution Plot')
                fig = plt.figure(figsize=(100, 40))
                df['Country Name'].value_counts().plot(kind='bar')
                st.pyplot(fig)

                st.header('3.Multiple Graphs')
                col1, col2 = st.columns(2)
                with col1:
                    col1.header = 'KDE = False'
                    fig1 = plt.figure(figsize=(8, 5))
                    sns.distplot(df['Value'], kde=False)
                    st.pyplot(fig1)
                with col2:
                    col2.header = 'Hist = False'
                    fig2 = plt.figure(figsize=(8, 5))
                    sns.distplot(df['Value'], hist=False)
                    st.pyplot(fig2)

                st.header('4.changing styles in multiple graphs')
                col1, col2 = st.columns(2)
                with col1:
                    # col1.header = 'KDE = False'
                    fig = plt.figure()
                    sns.set_style('darkgrid')
                    sns.set_context('notebook')
                    sns.distplot(df['Value'], hist=False)
                    st.pyplot(fig)
                with col2:
                    fig = plt.figure()
                    # col2.header = 'Hist = False'
                    sns.set_theme(context='poster', style='darkgrid')
                    # fig2 = plt.figure(figsize = (8,5))
                    sns.distplot(df['Value'], hist=False)
                    st.pyplot(fig)

                st.subheader('5.COUNT PLOT')
                fig = plt.figure(figsize=(100, 40))
                sns.countplot(data=df, x='Country Name')
                st.pyplot(fig)

                st.subheader('6.BOX PLOT')
                fig = plt.figure(figsize=(100, 40))
                sns.boxplot(data=df, x='Country Name', y='Value')
                st.pyplot(fig)

                st.subheader('7.VIOLIN PLOT')
                fig = plt.figure(figsize=(100, 40))
                sns.violinplot(data=df, x='Country Name', y='Value')
                st.pyplot(fig)

                st.subheader('8.FINDING ARAB WORLD DATA')
                df_pr = df[df['Country Name'] == 'Arab World']
                df_pr
                data = df_pr.values
                st.subheader('9.CALUCLATE GDP CHANGE of arab world')
                gdp_change = [0]
                for i in range(1, len(data)):
                    prev = data[i - 1][3]
                    cur = data[i][3]

                    gdp_change.append(round(((cur - prev) / prev) * 100, 2))

                df_pr = df_pr.assign(GDP=gdp_change)
                df_pr
                df_pr = df[df["Country Name"] == 'Arab World']

                # bad = df_pr.plot(kind = 'line', x = 'Year', y = 'Value',
                #            title='Graph of arab world',
                #            figsize = (15,10),
                #            grid = True,
                #            legend = True,
                #            ylabel = 'GDP',
                #            xlabel = 'YEARS')
                # bad

                # st.area_chart(df,x = 'Country Name',y = 'Country Code')

                df_pr1 = df[df['Country Name'] == 'World']
                fg = px.line(df_pr1, x='Year', y='Value', title='World GDP Analysis')
                fg

                df_pr2 = df[df['Country Name'] == 'India']
                fg = px.line(df_pr2, x='Year', y='Value', title='India GDP Analysis')
                fg

                df_pr3 = df[df['Country Name'] == 'Euro area']
                fg = px.line(df_pr3, x='Year', y='Value', title='Euro area GDP Analysis')
                fg

                df_pr4 = df[df['Country Name'] == 'China']
                fg = px.line(df_pr4, x='Year', y='Value', title='China GDP Analysis')
                fg

                st.subheader("for ARABWORLD")
                st.area_chart(df_pr, x="Value", y=["Year"], color=["#0000FF"]  # Optional
                              )


    elif choice == "india":
        st.subheader("Perform Exploratory data Analysis with Pandas Profiling Library")
        # data_file = st.file_uploader("Upload a csv file", type=["csv"])
        # if st.button("Analyze"):
        #     if data_file is not None:
        #         # Pandas Profiling Report
        #         @st.cache
        #         def load_csv():
        #             csv = pd.read_csv('Data IA.csv')
        #
        #             return csv
        #
        #         Data1A  = load_csv()
        data_file = st.file_uploader("Upload a csv file", type=["csv"])
        if st.button("Analyze"):
            if data_file is not None:
                data = pd.read_csv(data_file)
                st.dataframe(data)
                c = data.head()
                st.write(c)
                @st.cache_data
                def load_data():
                    csv  = pd.read_csv('Data IA.csv')

                    return csv
                Data1A = load_data()
                new_df = Data1A

                st.subheader('INTERACTIVE CHART')
                state_name = new_df.columns.tolist()
                state_choices = st.multiselect('choose state', state_name)
                new_df1 = new_df[state_choices]
                st.line_chart(new_df1)

                st.subheader('Remove trailing spaces from column headers like Andhra Pradesh  to Andhra Pradesh')
                Data1A.rename(columns=lambda x: x.strip(), inplace=True)
                Data1A

                st.subheader('removing WestBengal data is insufficient,it can be ignored')
                Data1A.drop(['West Bengal1'], axis=1, inplace=True)
                Data1A

                st.subheader('Set index as Duration column')
                Data1A.set_index(['Duration'], inplace=True)
                Data1A

                st.subheader('removing 2016-17 rows as data is insufficient can be ignored')
                Data1A.drop(['2016-17'], inplace=True)

                st.subheader('replacing Nan with 0')
                Data1A.fillna(0, inplace=True)

                Data1A

                st.subheader('Create DataFrame Growth_Data1A with data of Growth Percentage')
                Growth_Data1A = Data1A[Data1A["Items  Description"] == '(% Growth over previous year)']
                Growth_Data1A

                st.subheader('Plotting linecharts to visualize GSDP and Assumption - Ignoring J&K as its a U.T. now')

                st.title('Andhra Pradesh')
                fig = st.line_chart(data=Growth_Data1A, x='Andhra Pradesh', y='All_India GDP')

                st.title('Arunachal Pradesh')
                fig = st.line_chart(data=Growth_Data1A, x='Arunachal Pradesh', y='All_India GDP')

                st.title('Assam')
                fig = st.line_chart(data=Growth_Data1A, x='Assam', y='All_India GDP')

                st.title('Bihar')
                fig = st.line_chart(data=Growth_Data1A, x='Bihar', y='All_India GDP')

                st.title('Chhattisgarh')
                fig = st.line_chart(data=Growth_Data1A, x='Chhattisgarh', y='All_India GDP')

                st.title('Gujarat')
                fig = st.line_chart(data=Growth_Data1A, x='Gujarat', y='All_India GDP')

                st.title('Haryana')
                fig = st.line_chart(data=Growth_Data1A, x='Haryana', y='All_India GDP')

                st.title('Himachal Pradesh')
                fig = st.line_chart(data=Growth_Data1A, x='Himachal Pradesh', y='All_India GDP')

                # Ignoring J&K as its a U.T. now
                st.title('Jammu & Kashmir')
                fig = st.line_chart(data=Growth_Data1A, x='Jammu & Kashmir', y='All_India GDP')

                st.title('Jharkhand')
                fig = st.line_chart(data=Growth_Data1A, x='Jharkhand', y='All_India GDP')

                st.title('Karnataka')
                fig = st.line_chart(data=Growth_Data1A, x='Karnataka', y='All_India GDP')

                st.title('Kerala')
                fig = st.line_chart(data=Growth_Data1A, x='Kerala', y='All_India GDP')

                st.title('Madhya Pradesh')
                fig = st.line_chart(data=Growth_Data1A, x='Madhya Pradesh', y='All_India GDP')

                st.title('Maharashtra')
                fig = st.line_chart(data=Growth_Data1A, x='Maharashtra', y='All_India GDP')

                st.title('Manipur')
                fig = st.line_chart(data=Growth_Data1A, x='Manipur', y='All_India GDP')

                st.title('Meghalaya')
                fig = st.line_chart(data=Growth_Data1A, x='Meghalaya', y='All_India GDP')

                st.title('Mizoram')
                fig = st.line_chart(data=Growth_Data1A, x='Mizoram', y='All_India GDP')

                st.title('Nagaland')
                fig = st.line_chart(data=Growth_Data1A, x='Nagaland', y='All_India GDP')

                st.title('Odisha')
                fig = st.line_chart(data=Growth_Data1A, x='Odisha', y='All_India GDP')

                st.title('Punjab')
                fig = st.line_chart(data=Growth_Data1A, x='Punjab', y='All_India GDP')

                st.title('Rajasthan')
                fig = st.line_chart(data=Growth_Data1A, x='Rajasthan', y='All_India GDP')

                st.title('Sikkim')
                fig = st.line_chart(data=Growth_Data1A, x='Sikkim', y='All_India GDP')

                st.title('Tamil Nadu')
                fig = st.line_chart(data=Growth_Data1A, x='Tamil Nadu', y='All_India GDP')

                st.title('Telangana')
                fig = st.line_chart(data=Growth_Data1A, x='Telangana', y='All_India GDP')

                st.title('Tripura')
                fig = st.line_chart(data=Growth_Data1A, x='Tripura', y='All_India GDP')

                st.title('Uttar Pradesh')
                fig = st.line_chart(data=Growth_Data1A, x='Uttar Pradesh', y='All_India GDP')

                st.title('Uttarakhand')
                fig = st.line_chart(data=Growth_Data1A, x='Uttarakhand', y='All_India GDP')

                st.title('Goa')
                fig = st.line_chart(data=Growth_Data1A, x='Goa', y='All_India GDP')

                st.subheader('line graph for the nation')
                st.title('INDIA')
                fig = st.line_chart(data=Growth_Data1A, x='Delhi', y='All_India GDP')
                # st.pyplot(fig)
                fig = st.area_chart(Growth_Data1A, x='Delhi', y='All_India GDP')

                st.text("What is the Nation's growth rate?")
                st.text("India's growth rate during 2015-16 is 9.99%")
                st.text("India's growth rate during 2011-17 is 74.57%")
                # KA_India = Growth_Data1A[['Karnataka', 'All_India GDP']]
                # cor = KA_India.corr()
                # st.subheader('Correlation Heatmap')
                # fig = sns.heatmap(cor, annot=True)
                # st.pyplot(fig)

                # Data1A = pd.read_csv('Data IA.csv')
                pd.options.mode.chained_assignment = None
                GDP_Data = Data1A[(Data1A["Items  Description"] == 'GSDP - CURRENT PRICES (` in Crore)') & (
                            Data1A["Duration"] == '2015-16')]
                # Dropping unwanted columns
                GDP_Data.drop(['All_India GDP'], axis=1, inplace=True)
                GDP_Data.drop(['Duration'], axis=1, inplace=True)
                GDP_Data.drop(['Items  Description'], axis=1, inplace=True)
                # Removing Union Territories
                GDP_Data.drop(['Chandigarh'], axis=1, inplace=True)
                GDP_Data.drop(['Puducherry'], axis=1, inplace=True)
                GDP_Data.drop(['Jammu & Kashmir'], axis=1, inplace=True)
                GDP_Data.drop(['Delhi'], axis=1, inplace=True)

                # Drop Nan as 0
                GDP_Data.dropna(axis=1, inplace=True)
                # Transposing data for easier plots
                GDP_Data = GDP_Data.T.reset_index()
                # Adding column names
                GDP_Data.columns = ["States", "GDSP"]
                # Sorting data ascending
                GDP_Data = GDP_Data.sort_values(by='GDSP', ascending=True)
                GDP_Data
                # GDP_Data.plot.barh(x = "States", y = "GDSP", figsize=(20,20))
                # plt.tight_layout()

                st.subheader('Bottom 5 states based on GDP')
                GDP_Data[0:5]
                st.subheader('Top 5 states based on GDP')
                GDP_Data[-5:]

    else:
        pass


if __name__ == "__main__":
    main()