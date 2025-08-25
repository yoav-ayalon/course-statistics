# course-statistics

This project analyzes a dataset of insurance claims and driver statistics using Python. The main goals are to explore relationships between demographic, socioeconomic, and driving variables, and to build predictive models for insurance claim outcomes.

## Contents

- `CI_stat.csv`: The main dataset containing driver and vehicle statistics.
- `Project.py`: Python code for data analysis, visualization, and modeling.
- `project report.pdf`: Detailed report describing the analysis, results, and conclusions.

## Methods Used

### Data Preparation
- **Mapping categorical variables**: Converts string categories (e.g., education, income, gender) to numerical codes for analysis.
- **Feature engineering**: Creates new variables such as risk index, economic index, and interactions between features.

### Exploratory Data Analysis
- **Visualization**: Uses seaborn and matplotlib for histograms, boxplots, violin plots, and scatter plots to explore distributions and relationships.
- **Correlation analysis**: Computes and visualizes correlation matrices for numerical features.
- **Statistical tests**: Performs ANOVA and chi-square tests to examine relationships between categorical variables and outcomes.

### Modeling
- **Linear regression**: Fits models to predict continuous outcomes and checks assumptions (normality, homoscedasticity).
- **Logistic regression**: Builds models to predict binary insurance claim outcomes, including stepwise feature selection based on BIC and p-values.
- **Model comparison**: Compares full and reduced models using metrics like AIC, BIC, pseudo R², and likelihood ratio tests.
- **Performance evaluation**: Plots ROC and precision-recall curves to assess classification performance.

## How to Run

1. Ensure you have Python and required libraries (`pandas`, `numpy`, `seaborn`, `matplotlib`, `scipy`, `statsmodels`, `sklearn`) installed.
2. Place all files in the same directory.
3. Run `Project.py` to execute the analysis and view results.

## Report

See [`project report.pdf`](project%20report.pdf) for a detailed explanation of the methodology, results, and interpretations.
