import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set page configuration for layout and styling
st.set_page_config(
    page_title="Student Performance & Traffic Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set matplotlib style and consistent color palette
sns.set_theme(style="whitegrid")
sns.set_palette("viridis")

# Custom Title and Description
st.title("📊 Student Performance & Traffic Statistical Analysis")
st.markdown("""
This interactive application provides exploratory data analysis, correlation mapping, and inferential statistics for 
both Academic Performance and Urban Traffic datasets.
""")

# =========================================================================
# ADJUST COLUMN CONFIGURATIONS HERE ONCE YOU LOAD THE ACTUAL KAGGLE FILES
# Make sure these lists match the numeric headers of your downloaded CSVs.
# =========================================================================
DEFAULT_STUDENT_COLS = ['age', 'study_time', 'failures', 'absences', 'G1', 'G2', 'G3']
DEFAULT_TRAFFIC_COLS = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'traffic_volume', 'hour']

# Helper function to compute confidence intervals
def compute_ci(data, confidence_level=0.95):
    """
    Computes t-distribution confidence interval for the mean.
    Returns: (mean, lower_bound, upper_bound, margin_of_error)
    """
    clean_data = data.dropna()
    n = len(clean_data)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan
    mean = np.mean(clean_data)
    sem = stats.sem(clean_data)
    ci = stats.t.interval(confidence_level, df=n-1, loc=mean, scale=sem)
    moe = sem * stats.t.ppf((1 + confidence_level) / 2., n-1)
    return mean, ci[0], ci[1], moe

# Create tabs for the two datasets
tab1, tab2 = st.tabs(["🏫 Student Performance", "🚗 Traffic Analysis"])

# -------------------------------------------------------------------------
# TAB 1: Student Performance
# -------------------------------------------------------------------------
with tab1:
    st.header("Student Academic Performance Analysis")
    
    # Sidebar setup for files
    st.sidebar.markdown("### 🏫 Student Performance Data")
    uploaded_student = st.sidebar.file_uploader(
        "Upload custom Student CSV:", 
        type=["csv"], 
        key="student_uploader"
    )
    
    # Load dataset logic
    student_df = None
    if uploaded_student is not None:
        try:
            student_df = pd.read_csv(uploaded_student)
            st.sidebar.success("Custom Student CSV loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
    else:
        student_csv_path = os.path.join("data", "student_performance.csv")
        if os.path.exists(student_csv_path):
            student_df = pd.read_csv(student_csv_path)
            st.sidebar.info("Using default student mock dataset.")
        else:
            st.error("Default dataset not found. Please upload a student performance CSV in the sidebar.")
            
    if student_df is not None:
        # Detect numeric columns dynamically
        available_cols = [col for col in student_df.columns if pd.api.types.is_numeric_dtype(student_df[col])]
        
        if len(available_cols) == 0:
            st.error("No numeric columns found in the loaded dataset. Cannot perform statistical calculations.")
        else:
            # Set default selections based on availability
            default_selections = [c for c in DEFAULT_STUDENT_COLS if c in available_cols]
            if not default_selections:
                default_selections = available_cols[:min(4, len(available_cols))]
                
            selected_student_cols = st.sidebar.multiselect(
                "Select Student Columns for Correlation:",
                options=available_cols,
                default=default_selections,
                key="student_col_multiselect"
            )
            
            # Main Layout: Side-by-side stats and plots
            col1, col2 = st.columns([1, 1.2])
            
            with col1:
                st.subheader("Descriptive Statistics")
                if len(selected_student_cols) > 0:
                    desc_df = student_df[selected_student_cols].describe().T
                    # Compute skewness
                    skewness = student_df[selected_student_cols].apply(lambda x: stats.skew(x.dropna()))
                    desc_df['skewness'] = skewness
                    st.dataframe(desc_df.style.format(precision=3), use_container_width=True)
                else:
                    st.warning("Please select at least one column to view statistics.")
                    
                # Confidence Interval Calculator Widget
                st.subheader("🔮 Mean Confidence Interval Calculator")
                ci_col = st.selectbox("Select variable for CI:", options=available_cols, key="student_ci_var")
                ci_level_pct = st.selectbox("Select Confidence Level:", options=["90%", "95%", "99%"], index=1, key="student_ci_level")
                
                conf_map = {"90%": 0.90, "95%": 0.95, "99%": 0.99}
                level = conf_map[ci_level_pct]
                
                if ci_col:
                    mean, lower, upper, moe = compute_ci(student_df[ci_col], level)
                    if not np.isnan(mean):
                        st.metric(label=f"Sample Mean ({ci_col})", value=f"{mean:.3f}")
                        st.success(f"**{ci_level_pct} Confidence Interval:** ({lower:.3f}, {upper:.3f})")
                        st.info(f"**Margin of Error:** {moe:.4f}  \n**Interpretation:** Under repeated sampling, {ci_level_pct} of intervals calculated in this manner will contain the true population mean of **{ci_col}**.")
                    else:
                        st.error("Insufficient non-null data to compute confidence intervals.")
                        
            with col2:
                st.subheader("Correlation Heatmap")
                if len(selected_student_cols) > 1:
                    fig, ax = plt.subplots(figsize=(6, 4.5))
                    corr_matrix = student_df[selected_student_cols].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", ax=ax)
                    plt.title("Pearson Correlation Coefficients", pad=12)
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("Select at least 2 columns in the sidebar to display the correlation heatmap.")
                    
                # Distribution Plot Selector
                st.subheader("Univariate Distribution Plot")
                dist_col = st.selectbox("Select variable to plot distribution:", options=available_cols, key="student_dist_var")
                if dist_col:
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    sns.histplot(student_df[dist_col], kde=True, ax=ax)
                    ax.set_title(f"Distribution & Density of {dist_col}")
                    st.pyplot(fig)
                    plt.close()

            # New Visualization Selector per tab
            st.markdown("---")
            st.subheader("📊 Advanced Exploratory Plots")
            
            student_visual_options = [
                "Select a chart type...",
                "Study Time vs Final Grade (Box Plot)",
                "School Support vs Final Grade (Violin Plot)",
                "Parental Education vs Final Grade (Bar Chart)",
                "Strongest Correlated Pair (Scatter Plot with Regression Line)",
                "Multi-Feature Matrix Colored by Outcome (Pair Plot)"
            ]
            selected_student_visual = st.selectbox("Select Visual Type:", options=student_visual_options, key="student_visual_select")
            
            if selected_student_visual == "Study Time vs Final Grade (Box Plot)":
                if 'study_time' in student_df.columns and 'G3' in student_df.columns:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.boxplot(data=student_df, x='study_time', y='G3', hue='study_time', legend=False, ax=ax)
                    ax.set_title("Final Grade Distribution Grouped by Weekly Study Time Bucket", fontsize=12)
                    ax.set_xlabel("Study Time Bracket (1 = Low, 4 = High)")
                    ax.set_ylabel("Final Grade (G3)")
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.warning("Required columns ('study_time', 'G3') not found in the loaded student dataset.")
                    
            elif selected_student_visual == "School Support vs Final Grade (Violin Plot)":
                if 'schoolsup' in student_df.columns and 'G3' in student_df.columns:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.violinplot(data=student_df, x='schoolsup', y='G3', hue='schoolsup', legend=False, ax=ax)
                    ax.set_title("Final Grade Density Split by School Support Status", fontsize=12)
                    ax.set_xlabel("Received Extra School Support (yes/no)")
                    ax.set_ylabel("Final Grade (G3)")
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.warning("Required columns ('schoolsup', 'G3') not found in the loaded student dataset.")
                    
            elif selected_student_visual == "Parental Education vs Final Grade (Bar Chart)":
                if 'parent_education' in student_df.columns and 'G3' in student_df.columns:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    # Check what values exist in parent_education to order properly
                    order_list = ['none', 'primary', 'secondary', 'higher']
                    order_filtered = [o for o in order_list if o in student_df['parent_education'].unique()]
                    if not order_filtered:
                        order_filtered = None
                    sns.barplot(data=student_df, x='parent_education', y='G3', errorbar=('ci', 95),
                                order=order_filtered, hue='parent_education', legend=False, ax=ax)
                    ax.set_title("Mean Final Grade by Parental Education Level (95% CI)", fontsize=12)
                    ax.set_xlabel("Highest Parental Education Level")
                    ax.set_ylabel("Average Final Grade (G3)")
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.warning("Required columns ('parent_education', 'G3') not found in the loaded student dataset.")
                    
            elif selected_student_visual == "Strongest Correlated Pair (Scatter Plot with Regression Line)":
                # Find strongest correlated pair among numeric columns (excluding self-correlation)
                if len(available_cols) >= 2:
                    corr_matrix = student_df[available_cols].corr().abs()
                    # Mask diagonal
                    np.fill_diagonal(corr_matrix.values, 0)
                    # Find coordinates of max correlation
                    max_corr_idx = corr_matrix.stack().idxmax()
                    x_col, y_col = max_corr_idx
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    r_val, p_val = stats.pearsonr(student_df[x_col], student_df[y_col])
                    sns.regplot(data=student_df, x=x_col, y=y_col, scatter_kws={'alpha':0.6}, line_kws={'color':'red', 'linewidth':2}, ax=ax)
                    ax.set_title(f"OLS Regression: {x_col} vs. {y_col}", fontsize=12)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    
                    # Annotate correlation coefficient
                    ax.text(0.05, 0.95, f"r = {r_val:.3f}\np-value = {p_val:.3e}", 
                            transform=ax.transAxes, verticalalignment='top',
                            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.warning("Insufficient numeric variables to determine correlated pairs.")
                    
            elif selected_student_visual == "Multi-Feature Matrix Colored by Outcome (Pair Plot)":
                if 'G3' in student_df.columns:
                    # Define outcome status
                    student_df['status'] = np.where(student_df['G3'] >= 10, 'Pass', 'Fail')
                    # Select cols
                    cols_to_plot = [c for c in ['study_time', 'G1', 'G2', 'G3'] if c in student_df.columns]
                    if len(cols_to_plot) >= 2:
                        cols_to_plot_with_status = cols_to_plot + ['status']
                        fig = sns.pairplot(student_df[cols_to_plot_with_status], hue='status', diag_kind="kde", plot_kws={'alpha': 0.75})
                        st.pyplot(fig)
                    else:
                        st.warning("Insufficient grade columns to plot pair plot.")
                else:
                    st.warning("Required column 'G3' not found to calculate pass/fail status.")

            # Written Summary / Insight Placeholders
            st.markdown("---")
            st.subheader("📝 Analytical Summary & Key Insights")
            st.info("**[INSTRUCTIONS]**: Update this summary section with your own findings after you download and load the real Kaggle student performance dataset.")
            
            st.markdown("""
            *   **Insight 1 (EXAMPLE)**: The final grade (`G3`) shows a strong positive correlation with previous assessment periods (`G1` and `G2`), indicating that earlier performance is a reliable indicator of final outcome. *(Update with actual Pearson coefficient once real data is loaded)*.
            *   **Insight 2 (EXAMPLE)**: Weekly study time (`study_time`) has a positive association with performance variables, while previous class failures (`failures`) show a negative correlation.
            *   **Insight 3 (EXAMPLE)**: Student absences (`absences`) display a highly positive skew, meaning a majority of students have low absence counts, but a few outliers have extremely high counts.
            """)

# -------------------------------------------------------------------------
# TAB 2: Traffic Analysis
# -------------------------------------------------------------------------
with tab2:
    st.header("Urban Traffic Volume Analysis")
    
    # Sidebar setup for traffic files
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚗 Traffic Flow Data")
    uploaded_traffic = st.sidebar.file_uploader(
        "Upload custom Traffic CSV:", 
        type=["csv"], 
        key="traffic_uploader"
    )
    
    # Load dataset logic
    traffic_df = None
    if uploaded_traffic is not None:
        try:
            traffic_df = pd.read_csv(uploaded_traffic)
            st.sidebar.success("Custom Traffic CSV loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
    else:
        traffic_csv_path = os.path.join("data", "traffic_volume.csv")
        if os.path.exists(traffic_csv_path):
            traffic_df = pd.read_csv(traffic_csv_path)
            st.sidebar.info("Using default traffic mock dataset.")
        else:
            st.error("Default dataset not found. Please upload a traffic volume CSV in the sidebar.")
            
    if traffic_df is not None:
        # Detect numeric columns dynamically
        traffic_available_cols = [col for col in traffic_df.columns if pd.api.types.is_numeric_dtype(traffic_df[col])]
        
        if len(traffic_available_cols) == 0:
            st.error("No numeric columns found in the loaded dataset. Cannot perform statistical calculations.")
        else:
            # Set default selections based on availability
            default_traffic_selections = [c for c in DEFAULT_TRAFFIC_COLS if c in traffic_available_cols]
            if not default_traffic_selections:
                default_traffic_selections = traffic_available_cols[:min(4, len(traffic_available_cols))]
                
            selected_traffic_cols = st.sidebar.multiselect(
                "Select Traffic Columns for Correlation:",
                options=traffic_available_cols,
                default=default_traffic_selections,
                key="traffic_col_multiselect"
            )
            
            # Main Layout: Side-by-side stats and plots
            col1, col2 = st.columns([1, 1.2])
            
            with col1:
                st.subheader("Descriptive Statistics")
                if len(selected_traffic_cols) > 0:
                    traffic_desc_df = traffic_df[selected_traffic_cols].describe().T
                    # Compute skewness
                    traffic_skewness = traffic_df[selected_traffic_cols].apply(lambda x: stats.skew(x.dropna()))
                    traffic_desc_df['skewness'] = traffic_skewness
                    st.dataframe(traffic_desc_df.style.format(precision=3), use_container_width=True)
                else:
                    st.warning("Please select at least one column to view statistics.")
                    
                # Confidence Interval Calculator Widget
                st.subheader("🔮 Mean Confidence Interval Calculator")
                traffic_ci_col = st.selectbox("Select variable for CI:", options=traffic_available_cols, key="traffic_ci_var")
                traffic_ci_level_pct = st.selectbox("Select Confidence Level:", options=["90%", "95%", "99%"], index=1, key="traffic_ci_level")
                
                traffic_level = conf_map[traffic_ci_level_pct]
                
                if traffic_ci_col:
                    mean, lower, upper, moe = compute_ci(traffic_df[traffic_ci_col], traffic_level)
                    if not np.isnan(mean):
                        st.metric(label=f"Sample Mean ({traffic_ci_col})", value=f"{mean:.3f}")
                        st.success(f"**{traffic_ci_level_pct} Confidence Interval:** ({lower:.3f}, {upper:.3f})")
                        st.info(f"**Margin of Error:** {moe:.4f}  \n**Interpretation:** Under repeated sampling, {traffic_ci_level_pct} of intervals calculated in this manner will contain the true population mean of **{traffic_ci_col}**.")
                    else:
                        st.error("Insufficient non-null data to compute confidence intervals.")
                        
            with col2:
                st.subheader("Correlation Heatmap")
                if len(selected_traffic_cols) > 1:
                    fig, ax = plt.subplots(figsize=(6, 4.5))
                    traffic_corr_matrix = traffic_df[selected_traffic_cols].corr()
                    sns.heatmap(traffic_corr_matrix, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, fmt=".2f", ax=ax)
                    plt.title("Pearson Correlation Coefficients", pad=12)
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("Select at least 2 columns in the sidebar to display the correlation heatmap.")
                    
                # Distribution Plot Selector
                st.subheader("Univariate Distribution Plot")
                traffic_dist_col = st.selectbox("Select variable to plot distribution:", options=traffic_available_cols, key="traffic_dist_var")
                if traffic_dist_col:
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    sns.histplot(traffic_df[traffic_dist_col], kde=True, ax=ax)
                    ax.set_title(f"Distribution & Density of {traffic_dist_col}")
                    st.pyplot(fig)
                    plt.close()

            # New Traffic Visualization selector
            st.markdown("---")
            st.subheader("📊 Advanced Exploratory Plots")
            
            traffic_visual_options = [
                "Select a chart type...",
                "Traffic Volume by Weather (Box Plot)",
                "Traffic Volume: Holidays vs Non-Holidays (Bar Chart)",
                "Diurnal Calendar Grid (Heatmap)"
            ]
            selected_traffic_visual = st.selectbox("Select Visual Type:", options=traffic_visual_options, key="traffic_visual_select")
            
            if selected_traffic_visual == "Traffic Volume by Weather (Box Plot)":
                if 'weather_main' in traffic_df.columns and 'traffic_volume' in traffic_df.columns:
                    fig, ax = plt.subplots(figsize=(9, 5))
                    sns.boxplot(data=traffic_df, x='weather_main', y='traffic_volume', hue='weather_main', legend=False, ax=ax)
                    ax.set_title("Hourly Traffic Volume Distribution by Weather Condition", fontsize=12)
                    ax.set_xlabel("Primary Weather Classification")
                    ax.set_ylabel("Traffic Volume Count")
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.warning("Required columns ('weather_main', 'traffic_volume') not found in the loaded traffic dataset.")
                    
            elif selected_traffic_visual == "Traffic Volume: Holidays vs Non-Holidays (Bar Chart)":
                if 'holiday' in traffic_df.columns and 'traffic_volume' in traffic_df.columns:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    sns.barplot(data=traffic_df, x='holiday', y='traffic_volume', errorbar=('ci', 95), hue='holiday', legend=False, ax=ax)
                    ax.set_title("Comparison of Average Traffic Volume: Holidays vs. Non-Holidays", fontsize=12)
                    ax.set_xlabel("Day Classification (Holiday vs. Regular Day)")
                    ax.set_ylabel("Mean Traffic Volume Count (95% CI)")
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.warning("Required columns ('holiday', 'traffic_volume') not found in the loaded traffic dataset.")
                    
            elif selected_traffic_visual == "Diurnal Calendar Grid (Heatmap)":
                if 'date_time' in traffic_df.columns and 'traffic_volume' in traffic_df.columns and 'hour' in traffic_df.columns:
                    # Dynamically calculate day_name
                    traffic_df['day_name'] = pd.to_datetime(traffic_df['date_time']).dt.day_name()
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    
                    try:
                        traffic_pivot = traffic_df.pivot_table(values='traffic_volume', index='day_name', columns='hour', aggfunc='mean')
                        traffic_pivot = traffic_pivot.reindex(day_order)
                        
                        fig, ax = plt.subplots(figsize=(12, 5.5))
                        sns.heatmap(traffic_pivot, cmap='viridis', cbar_kws={'label': 'Mean Traffic Volume'}, ax=ax)
                        ax.set_title("Diurnal Calendar Grid Heatmap: Hourly Average Traffic Volume by Day of Week", fontsize=13)
                        ax.set_xlabel("Hour of Day (0-23)")
                        ax.set_ylabel("Day of Week")
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.error(f"Error drawing heatmap: {e}")
                else:
                    st.warning("Required columns ('date_time', 'traffic_volume', 'hour') not found in the loaded traffic dataset.")

            # Hourly volume line plot (original extra requirement)
            if 'hour' in traffic_df.columns and 'traffic_volume' in traffic_df.columns:
                st.markdown("---")
                st.subheader("Traffic Volume Hourly Trends")
                hourly_data = traffic_df.groupby('hour')['traffic_volume'].mean().reset_index()
                fig, ax = plt.subplots(figsize=(10, 3.5))
                sns.lineplot(data=hourly_data, x='hour', y='traffic_volume', color='#FF4D6D', marker='o', linewidth=2, ax=ax)
                ax.set_xticks(range(0, 24))
                ax.set_xlabel("Hour of Day (0-23)")
                ax.set_ylabel("Average Traffic Volume")
                ax.set_title("Diurnal Traffic Volume Pattern")
                st.pyplot(fig)
                plt.close()

            # Written Summary / Insight Placeholders
            st.markdown("---")
            st.subheader("📝 Analytical Summary & Key Insights")
            st.info("**[INSTRUCTIONS]**: Update this summary section with your own findings after you download and load the real Kaggle traffic volume dataset.")
            
            st.markdown("""
            *   **Insight 1 (EXAMPLE)**: Traffic volume peaks during standard weekday rush hours (around 7-8 AM and 4-6 PM), showing clear commuting cycles. *(Verify hourly trends once real dataset is loaded)*.
            *   **Insight 2 (EXAMPLE)**: Under extreme weather categories (e.g. high rain volume or snow), traffic speed/flow drops significantly, demonstrating safety-driven slowdowns.
            *   **Insight 3 (EXAMPLE)**: Temperature (`temp`) has a very weak correlation with traffic volume overall, but displays strong seasonal cycles.
            """)
st.sidebar.caption("EduMetrics Analytics App v2.2")
