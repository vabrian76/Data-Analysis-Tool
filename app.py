import streamlit as st
import base64
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ollama
from io import StringIO

# ----------------------------------
# Set Background Image from a Local File
# ----------------------------------
def set_background_from_local(image_path):
    try:
        with open(image_path, "rb") as image_file:
            img_bytes = image_file.read()
        encoded = base64.b64encode(img_bytes).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
                background-color: rgba(0, 0, 0, 0.7); /* Darker background */
                color: #000000; /* White text */
            }}
            .custom-title {{
                margin-top: -50px; /* Adjust this value to move the title upward */
                color: #000000; /* White text */
            }}
            .stMarkdown, .stButton>button {{
                color: #000000; /* White text */
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error setting background image: {e}")

# ----------------------------------
# Helper Functions for Data Analysis
# ----------------------------------
def load_data(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def clean_data(df):
    df = df.drop_duplicates()
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

def explore_data(df):
    st.subheader("Data Overview")
    st.write(df.head())
    
    st.subheader("Dataset Info")
    info_buffer = StringIO()
    df.info(buf=info_buffer)
    st.text(info_buffer.getvalue())
    
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
    st.subheader("Missing Values")
    st.write(df.isnull().sum())
    
    st.subheader("Correlation Matrix")
    st.write(df.corr())
    
    st.subheader("Pairplot")
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        sns.set(style="ticks")
        fig = sns.pairplot(numeric_df)
        st.pyplot(fig)
    else:
        st.info("No numeric columns available for pairplot.")

def analyze_with_ollama(df):
    prompt = f"""
    Here is a summary of the dataset:
    {df.describe().to_string()}
    
    Based on this data summary, provide insights on trends, anomalies, and possible predictions.
    """
    st.subheader("Ollama Analysis")
    try:
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        st.write(response["message"]["content"])
    except Exception as e:
        st.error(f"Error during Ollama analysis: {e}")

def null_percentage_analysis(df):
    st.subheader("Null Percentage Analysis")
    grouping_options = st.multiselect("Select grouping columns (optional):", options=df.columns.tolist(), default=[])
    features = st.multiselect("Select features to analyze for null percentages:", options=df.columns.tolist(), default=df.columns.tolist())
    if not features:
        st.info("Please select at least one feature to analyze.")
        return
    if grouping_options:
        try:
            null_percentage_df = df.groupby(grouping_options)[features].apply(lambda x: x.isnull().mean() * 100)
            st.write(null_percentage_df)
        except Exception as e:
            st.error(f"Error during grouped null percentage analysis: {e}")
    else:
        null_percentage = df[features].isnull().mean() * 100
        st.write(null_percentage.to_frame(name="Null Percentage"))

def column_view_indicator(df):
    st.subheader("Column View Indicator")
    column = st.selectbox("Select a column to view:", df.columns.tolist())
    view_indicator = st.selectbox("Select a view indicator:", ["Summary Statistics", "Null Percentage", "Distribution Plot", "Box Plot", "Value Counts"])
    if view_indicator == "Summary Statistics":
        st.write(df[column].describe())
    elif view_indicator == "Null Percentage":
        null_percentage = df[column].isnull().mean() * 100
        st.write(f"Null Percentage for {column}: {null_percentage:.2f}%")
    elif view_indicator == "Distribution Plot":
        if pd.api.types.is_numeric_dtype(df[column]):
            fig, ax = plt.subplots()
            sns.histplot(df[column].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {column}")
            st.pyplot(fig)
        else:
            st.warning("Distribution plot is only available for numeric columns.")
    elif view_indicator == "Box Plot":
        if pd.api.types.is_numeric_dtype(df[column]):
            fig, ax = plt.subplots()
            sns.boxplot(x=df[column].dropna(), ax=ax)
            ax.set_title(f"Box Plot of {column}")
            st.pyplot(fig)
        else:
            st.warning("Box plot is only available for numeric columns.")
    elif view_indicator == "Value Counts":
        st.write(df[column].value_counts())

def filter_by_view_indicator(df):
    st.subheader("Filter Data by View Indicator")
    filter_column = st.selectbox("Select a column for filtering:", df.columns.tolist())
    unique_values = df[filter_column].dropna().unique().tolist()
    if unique_values:
        selected_values = st.multiselect(f"Select values in '{filter_column}' to filter the data:", options=unique_values, default=unique_values)
        if selected_values:
            filtered_df = df[df[filter_column].isin(selected_values)]
            st.write(f"Filtered Data based on {filter_column}:", filtered_df)
            return filtered_df
        else:
            st.info("No values selected. Showing the full dataset.")
            return df
    else:
        st.info("No unique values found in this column.")
        return df

# ----------------------------------
# Main Analysis Page
# ----------------------------------
def show_analysis_page():
    st.title("V and D's Polaris Data Analysis Tool")
    st.sidebar.header("Upload and Analysis Options")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    analysis_options = st.sidebar.multiselect(
        "Select Analysis Options:",
        ["Show Raw Data", "Clean Data", "Exploratory Data Analysis", "Ollama Analysis", "Null Percentage Analysis", "Column View Indicator", "Filter by View Indicator"],
        default=["Show Raw Data", "Exploratory Data Analysis", "Null Percentage Analysis", "Column View Indicator", "Filter by View Indicator"]
    )
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            if "Show Raw Data" in analysis_options:
                st.subheader("Raw Data")
                st.write(df)
            if "Clean Data" in analysis_options:
                st.subheader("Cleaned Data")
                df = clean_data(df)
                st.write(df)
            if "Filter by View Indicator" in analysis_options:
                df = filter_by_view_indicator(df)
            if "Exploratory Data Analysis" in analysis_options:
                explore_data(df)
            if "Ollama Analysis" in analysis_options:
                analyze_with_ollama(df)
            if "Null Percentage Analysis" in analysis_options:
                null_percentage_analysis(df)
            if "Column View Indicator" in analysis_options:
                column_view_indicator(df)
    else:
        st.info("Please upload a CSV file to begin analysis.")

# ----------------------------------
# Landing Page
# ----------------------------------
def show_landing_page():
    st.markdown('<div class="custom-title"><h1>Welcome to V and D\'s Polaris Data Analysis Tool</h1></div>', unsafe_allow_html=True)
    # st.write("Click *Start Analysis* to proceed to the analysis page.")
    if st.button("Start Analysis"):
        st.session_state["analysis_started"] = True
        st.rerun()

# ----------------------------------
# App Control
# ----------------------------------
def main():
    if "analysis_started" not in st.session_state:
        st.session_state["analysis_started"] = False

    if not st.session_state["analysis_started"]:
        set_background_from_local("background.jpg")
        show_landing_page()
    else:
        show_analysis_page()

if __name__ == "__main__":
    main()