import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from patsy import dmatrices

from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc, precision_recall_curve

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Load the data ------------------------------------------------------
df = pd.read_csv("CI_stat.csv")
# print(df.info())
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Defs ---------------------------------------------------------------
def map_data(df):
    # education_map = {"none": 1, "high school": 2, "university": 3}
    # income_map = {"poverty": 1, "working class": 2, "middle class": 3, "upper class": 4}
    # gender_map = {"male": 0, "female": 1}
    # postal_map = {10238: 1, 21217: 2, 32765: 3, 92101: 4}
    # type_map = {"sedan": 0, "sports car": 1}
    # year_map = {"before 2015": 0, "after 2015": 1}
    #
    # df["EDUCATION"] = df["EDUCATION"].map(education_map)
    # df["INCOME"] = df["INCOME"].map(income_map)
    # df["GENDER"] = df["GENDER"].map(gender_map)
    # df["POSTAL_CODE"] = df["POSTAL_CODE"].map(postal_map)
    # df["VEHICLE_TYPE"] = df["VEHICLE_TYPE"].map(type_map)
    # df["VEHICLE_YEAR"] = df["VEHICLE_YEAR"].map(year_map

    # Convert binary numerical columns (0/1) to boolean before dummy conversion
    binary_cols = ["VEHICLE_OWNERSHIP", "MARRIED", "CHILDREN"]
    df[binary_cols] = df[binary_cols].astype(bool)  # Convert to True/False

    # Identify categorical columns (excluding numerical ones)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Include POSTAL_CODE in categorical columns since it is not ordinal
    categorical_cols.append("POSTAL_CODE")

    # Convert categorical and binary boolean variables to dummy variables
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Convert boolean columns back to 0/1 after dummy transformation
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def change_driving_exprince(df):
    df["AGE_START_DRIVING"] = df["AGE"] - df["DRIVING_EXPERIENCE"]
    # # before change
    # plt.figure(figsize=(8, 5))
    # sns.histplot(df["AGE_START_DRIVING"], bins=20, kde=True)
    # plt.xlabel("Age When Started Driving")
    # plt.ylabel("Count")
    # plt.title("Distribution of Starting Driving Age")
    # plt.show()

    df.loc[df["AGE_START_DRIVING"] < 16, "DRIVING_EXPERIENCE"] = df["AGE"] - 16
    df["AGE_START_DRIVING"] = df["AGE"] - df["DRIVING_EXPERIENCE"]

    # # after change
    # plt.figure(figsize=(8, 5))
    # sns.histplot(df["AGE_START_DRIVING"], bins=20, kde=True)
    # plt.xlabel("Age When Started Driving")
    # plt.ylabel("Count")
    # plt.title("Distribution of Starting Driving Age")
    # plt.show()


def visualize_variable(df):
    sns.set_style("whitegrid")

    # Define numerical and categorical features
    numerical_features = ["CREDIT_SCORE", "ANNUAL_MILEAGE", "AGE", "DRIVING_EXPERIENCE",
                          "SPEEDING_VIOLATIONS", "PAST_ACCIDENTS", "AGE_START_DRIVING"]

    categorical_features = ["GENDER", "EDUCATION", "INCOME", "VEHICLE_OWNERSHIP", "VEHICLE_YEAR",
                            "MARRIED", "CHILDREN", "POSTAL_CODE", "VEHICLE_TYPE"]

    # -------- Box Plots (Numerical vs Outcome) --------
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(numerical_features):
        plt.subplot(2, 4, i + 1)
        sns.boxplot(x=df["OUTCOME"], y=df[col])
        plt.title(f"Box Plot of {col} by Outcome")

    plt.tight_layout()
    plt.show()

    # -------- Histograms (Numerical Variables) --------
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(numerical_features):
        plt.subplot(2, 4, i + 1)
        sns.histplot(df[col], bins=15, kde=True, edgecolor='black')
        plt.title(f"Histogram of {col}")

    plt.suptitle("Histograms of Numerical Variables")
    plt.tight_layout()
    plt.show()

    # -------- Box Plots for Categorical Variables --------
    plt.figure(figsize=(12, 10))
    for i, col in enumerate(categorical_features):
        plt.subplot(3, 3, i + 1)
        ax = sns.countplot(x=df[col], hue=df["OUTCOME"], palette="coolwarm")

        # Add count labels on top of bars
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}",
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.title(f"Outcome Distribution by {col}")

    plt.tight_layout()
    plt.show()

    # -------- Plot for Outcome --------
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=df["OUTCOME"], palette="coolwarm")

    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.title("Count of Each Outcome")
    plt.xlabel("Outcome")
    plt.ylabel("Count")
    plt.show()


def analyze_postal_code_influence(df):
    print("\n--- Analyzing the Influence of Postal Code on Insurance Claims ---\n")
    # Chi-Square Test: Check dependence between POSTAL_CODE and OUTCOME
    contingency_table = pd.crosstab(df["POSTAL_CODE"], df["OUTCOME"])
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-Square Statistic: {chi2_stat:.3f}, p-value: {p_value:.5f}")
    if p_value < 0.05:
        print("Result: There is a statistically significant relationship between postal code and claim rate.")
    else:
        print("Result: No significant relationship between postal code and claim rate.\n")

    # ANOVA Test: Check if claim rates differ significantly between postal codes
    groups = [df[df["POSTAL_CODE"] == code]["OUTCOME"] for code in df["POSTAL_CODE"].unique()]
    anova_result = f_oneway(*groups)
    print(f"\nANOVA F-statistic: {anova_result.statistic:.3f}, p-value: {anova_result.pvalue:.5f}")
    if anova_result.pvalue < 0.05:
        print("Result: Significant differences in claim rates exist between postal codes.")
    else:
        print("Result: No significant differences in claim rates between postal codes.\n")


def visualize_variable_relationships(df):
    sns.set_style("whitegrid")

    # -------- Relationships Income & Education & Gemder & Credit score --------
    education_order = ["none", "high school", "university"]
    income_order = ["poverty", "working class", "middle class", "upper class"]
    # Create a pivot table to count occurrences of each combination
    income_education_table = df.groupby(["INCOME", "EDUCATION"]).size().unstack()
    # Reorder the rows and columns
    income_education_table2 = income_education_table.reindex(index=income_order, columns=education_order)
    print(income_education_table2)

    plt.figure(figsize=(10, 5))
    sns.violinplot(x=df["EDUCATION"], y=df["CREDIT_SCORE"], hue=df["INCOME"], split=True, palette="coolwarm")
    plt.title("CREDIT_SCORE by EDUCATION and INCOME")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df["GENDER"], y=df["CREDIT_SCORE"], hue=df["EDUCATION"], palette="coolwarm")
    plt.title("CREDIT_SCORE by GENDER and EDUCATION")
    plt.show()

    # -------- Relationships SPEEDING_VIOLATIONS by POSTAL_CODE and VEHICLE_OWNERSHIP --------
    plt.figure(figsize=(6, 5))
    sns.violinplot(x=df["POSTAL_CODE"], y=df["SPEEDING_VIOLATIONS"], hue=df["VEHICLE_OWNERSHIP"], split=True,
                   palette="coolwarm")
    plt.title("SPEEDING_VIOLATIONS by POSTAL_CODE and VEHICLE_OWNERSHIP")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.violinplot(x=df["POSTAL_CODE"], y=df["SPEEDING_VIOLATIONS"], hue=df["VEHICLE_YEAR"], split=True,
                   palette="coolwarm")
    plt.title("SPEEDING_VIOLATIONS by POSTAL_CODE and VEHICLE_YEAR")
    plt.show()

    # -------- Relationships ANNUALMILEAGE & CHILFREN & MERRIGE --------
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df["CHILDREN"], y=df["ANNUAL_MILEAGE"], hue=df["MARRIED"], palette="coolwarm")
    plt.title("Annual Mileage by CHILDREN and Marriage Status")
    plt.show()


def correletions(df):
    numerical_features = ["CREDIT_SCORE", "ANNUAL_MILEAGE", "AGE", "DRIVING_EXPERIENCE",
                          "SPEEDING_VIOLATIONS", "PAST_ACCIDENTS", "AGE_START_DRIVING"]

    # -------- Correlation Heatmap --------
    plt.figure(figsize=(10, 8))
    corr_matrix = df[numerical_features].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap of Numerical Features")
    plt.xticks(rotation=20)
    plt.yticks(rotation=50)
    plt.show()


def check_multicollinearity_and_relationships(df):
    numerical_features = ["CREDIT_SCORE", "ANNUAL_MILEAGE", "AGE", "DRIVING_EXPERIENCE",
                          "SPEEDING_VIOLATIONS", "PAST_ACCIDENTS"]
    categorical_features = ["INCOME", "EDUCATION", "MARRIED", "CHILDREN", "VEHICLE_TYPE", "POSTAL_CODE"]

    print("\n=== 1. Checking for Multicollinearity ===")
    # VIF Calculation
    X = df[numerical_features]
    X = sm.add_constant(X)  # Adding constant for VIF calculation
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    print("\nVariance Inflation Factor (VIF):")
    print(vif_data[vif_data["Feature"] != "const"])

    print("\n=== 2. Examining Relationships between Income, Credit Score, and Education ===")

    # ANOVA for categorical impact on Credit Score
    anova_results_credit = []
    for cat_feature in ["INCOME", "EDUCATION"]:
        groups = [df[df[cat_feature] == val]["CREDIT_SCORE"].dropna() for val in df[cat_feature].unique()]
        if len(groups) > 1:
            f_stat, p_value = stats.f_oneway(*groups)
            anova_results_credit.append([cat_feature, f_stat, p_value])

    df_anova_credit = pd.DataFrame(anova_results_credit, columns=["Feature", "F-Statistic", "P-Value"])
    print("\nANOVA Test Results for Credit Score:")
    print(df_anova_credit)

    # Spearman Correlation between Income, Education, and Credit Score
    spearman_corr_credit = df[["INCOME", "EDUCATION", "CREDIT_SCORE", "OUTCOME"]].corr(method='spearman')
    print("\nSpearman Correlation for Income, Education & Credit Score:")
    print(spearman_corr_credit)

    print("\n=== 3. Examining Relationships between Annual Mileage, Accidents, Violations, and Driving Experience ===")

    # ANOVA for categorical impact on Annual Mileage
    anova_results_mileage = []
    for cat_feature in ["MARRIED", "CHILDREN"]:
        groups = [df[df[cat_feature] == val]["ANNUAL_MILEAGE"].dropna() for val in df[cat_feature].unique()]
        if len(groups) > 1:
            f_stat, p_value = stats.f_oneway(*groups)
            anova_results_mileage.append([cat_feature, f_stat, p_value])

    df_anova_mileage = pd.DataFrame(anova_results_mileage, columns=["Feature", "F-Statistic", "P-Value"])
    print("\nANOVA Test Results for Annual Mileage:")
    print(df_anova_mileage)

    # Spearman Correlation between Annual Mileage, Accidents, Violations, and Experience
    spearman_corr_mileage = df[
        ["ANNUAL_MILEAGE", "PAST_ACCIDENTS", "SPEEDING_VIOLATIONS", "DRIVING_EXPERIENCE", "OUTCOME"]].corr(
        method='spearman')
    print("\nSpearman Correlation for Mileage, Accidents, Violations & Driving Experience:")
    print(spearman_corr_mileage)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Models -------------------------------------------------------------
def full_linear_model(df):
    X = df.drop(columns=["OUTCOME"])
    y = df["OUTCOME"]

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    print("\n=== Full Linear Model Summary ===")
    print(model.summary())


def feature_selection_linear_model(df):
    X = df.drop(columns=["OUTCOME"])
    y = df["OUTCOME"]

    X = sm.add_constant(X)

    included = []
    threshold_in = 0.05
    threshold_out = 0.1

    while True:
        changed = False
        # check to add varibels
        excluded = list(set(X.columns) - set(included))  # משתנים שלא נבחרו עדיין
        new_pval = pd.Series(index=excluded, dtype=float)  # נריץ בדיקות לכל המשתנים שלא בפנים
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(X[included + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]  # ערך p למשתנה חדש

        best_pval = new_pval.min() if not new_pval.empty else None
        if best_pval is not None and best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True

        # check to remove varibels
        model = sm.OLS(y, sm.add_constant(X[included])).fit()
        pvalues = model.pvalues.iloc[1:]  # לא כולל הקבוע (const)
        worst_pval = pvalues.max() if not pvalues.empty else None
        if worst_pval is not None and worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True

        if not changed:
            break

    final_model = sm.OLS(y, sm.add_constant(X[included])).fit()

    print("\n=== Stepwise Selection Linear Model Summary ===")
    print(final_model.summary())

    return final_model


def plot_fitted_vs_actual(df, model):
    """
    Creates a scatter plot comparing the actual vs. predicted values of the model.
    """

    actual = df["OUTCOME"]  # True values
    predicted = model.fittedvalues  # Predicted values

    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=actual, y=predicted, alpha=0.5)
    plt.plot([0, 1], [0, 1], linestyle="--", color="red", linewidth=2)  # 45-degree reference line

    plt.xlabel("Actual OUTCOME")
    plt.ylabel("Predicted OUTCOME")
    plt.title("Actual vs. Predicted Values")
    plt.show()


def check_regression_assumptions(df, model):
    """
    Checks linear regression assumptions:
    1. Normality of residuals - Q-Q Plot, Shapiro-Wilk Test
    2. Homoscedasticity - Residuals vs. Fitted plot, Breusch-Pagan test

    :param df: DataFrame of the dataset
    :param model: OLS regression model
    """

    residuals = model.resid  # Model residuals
    fitted_values = model.fittedvalues  # Predicted values

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Q-Q Plot
    sm.qqplot(residuals, line='s', ax=axes[0])
    axes[0].set_title("Normal Q-Q Plot")

    # Residuals vs Fitted Values
    axes[1].scatter(fitted_values, residuals, alpha=0.5)
    axes[1].axhline(0, color='red', linestyle='dashed', linewidth=1)
    axes[1].set_xlabel("Fitted Values")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residuals Plot")

    plt.tight_layout()
    plt.show()

    # Shapiro-Wilk test for normality
    shapiro_test = stats.shapiro(residuals)
    print(f"Shapiro-Wilk Test: W={shapiro_test.statistic:.3f}, p-value={shapiro_test.pvalue:.6f}")
    if shapiro_test.pvalue < 0.05:
        print("Residuals are not normally distributed (Reject H0)")
    else:
        print("No significant deviation from normality (Fail to reject H0)")

    # Breusch-Pagan test for homoscedasticity
    exog = model.model.exog  # Explanatory variables
    bp_test = het_breuschpagan(residuals, exog)
    print(f"Breusch-Pagan Test: χ²={bp_test[0]:.3f}, p-value={bp_test[1]:.6f}")
    if bp_test[1] < 0.05:
        print("Heteroscedasticity detected (Reject H0)")
    else:
        print("No significant heteroscedasticity (Fail to reject H0)")


def logistic_regression_analysis(df):
    # Drop ID if exists
    df = df.drop(columns=["ID"], errors='ignore')

    # Separate features and target
    X = df.drop(columns=["OUTCOME"])
    y = df["OUTCOME"]

    # Add constant for intercept
    X = sm.add_constant(X)

    # Fit Logistic Regression Model
    logit_model = sm.Logit(y, X).fit()

    # Compute pseudo R² (McFadden's R²)
    pseudo_r2 = 1 - (logit_model.llf / logit_model.llnull)

    # Print Model Summary
    print("=" * 50)
    print("LOGISTIC REGRESSION MODEL SUMMARY")
    print("=" * 50)
    print(logit_model.summary())
    print(f"Pseudo R² (McFadden's R²): {pseudo_r2:.4f}")
    print(f"AIC: {logit_model.aic:.4f}")
    print(f"BIC: {logit_model.bic:.4f}")

    return logit_model


def stepwise_selection_logistic_regression(df):
    # Drop ID if exists
    df = df.drop(columns=["ID"], errors='ignore')

    # Separate features and target
    X = df.drop(columns=["OUTCOME"])
    y = df["OUTCOME"]

    # Add constant for intercept
    X = sm.add_constant(X)

    # Start with constant model
    included = ["const"]
    threshold_in = 0.05  # Enter if p < 0.05
    threshold_out = 0.10  # Remove if p > 0.10
    best_bic = float("inf")  # Store best BIC

    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)

        # Try adding each excluded feature
        for new_column in excluded:
            try:
                model = sm.Logit(y, sm.add_constant(X[included + [new_column]])).fit(disp=0)
                new_pval[new_column] = model.pvalues[new_column]
            except np.linalg.LinAlgError:
                continue  # Skip singular matrix cases

        # Add best feature (if significant & improves BIC)
        best_pval = new_pval.min() if not new_pval.empty else None
        if best_pval is not None and best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            temp_model = sm.Logit(y, X[included + [best_feature]]).fit(disp=0)
            if temp_model.bic < best_bic:  # Only add if BIC improves
                included.append(best_feature)
                best_bic = temp_model.bic
                changed = True

        # Remove worst feature (if p > threshold & improves BIC)
        model = sm.Logit(y, X[included]).fit(disp=0)
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() if not pvalues.empty else None
        if worst_pval is not None and worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            temp_model = sm.Logit(y, X[included].drop(columns=[worst_feature])).fit(disp=0)
            if temp_model.bic < best_bic:  # Only remove if BIC improves
                included.remove(worst_feature)
                best_bic = temp_model.bic
                changed = True

        if not changed:
            break

    # Fit final model
    final_model = sm.Logit(y, X[included]).fit()
    print(final_model.summary())

    # Compute pseudo R² (McFadden's R²)
    pseudo_r2 = 1 - (final_model.llf / final_model.llnull)

    print(f"Pseudo R² (McFadden's R²): {pseudo_r2:.4f}")
    print(f"AIC: {final_model.aic:.4f}")
    print(f"BIC: {final_model.bic:.4f} (Best BIC during selection)")

    return final_model


def prepare_data1_logistic_regression(df):
    # Drop ID if exists
    df = df.drop(columns=["ID"], errors='ignore')

    # Merge postal codes (Keep 21217 separate, others as one category)
    df["POSTAL_CODE_OTHER"] = (df[["POSTAL_CODE_32765", "POSTAL_CODE_92101"]].sum(axis=1) > 0).astype(int)
    df = df.drop(columns=["POSTAL_CODE_32765", "POSTAL_CODE_92101"], errors='ignore')

    # Create a combined socioeconomic index (Education + Income)
    df["ECONOMIC_INDEX"] = (
            df["INCOME_poverty"] * 1 +
            df["INCOME_working class"] * 2 +
            df["INCOME_upper class"] * 4 +  # Middle class is base (implicitly 3)
            df["EDUCATION_none"] * 1 +
            df["EDUCATION_university"] * 3  # High school is base (implicitly 2)
    )
    df = df.drop(columns=["INCOME_poverty", "INCOME_working class", "INCOME_upper class",
                          "EDUCATION_none", "EDUCATION_university"], errors='ignore')

    # Create a risk index from past accidents and speeding violations
    df["RISK_INDEX"] = df["PAST_ACCIDENTS"] * 0.7 + df["SPEEDING_VIOLATIONS"] * 0.3
    df = df.drop(columns=["PAST_ACCIDENTS", "SPEEDING_VIOLATIONS"], errors='ignore')

    # # Interaction: Risk index and gender
    # df["RISK_GENDER_INTERACTION"] = df["RISK_INDEX"] * df["GENDER_male"]
    # # Interaction: Age and economic index
    # df["AGE_ECONOMIC_INTERACTION"] = df["AGE"] * df["ECONOMIC_INDEX"]
    # # Interaction: Vehicle Ownership and Economic Index
    # df["VEHICLE_ECONOMIC_INTERACTION"] = df["VEHICLE_OWNERSHIP"] * df["ECONOMIC_INDEX"]

    # Interaction: Risk Index and Postal Code
    df["RISK_POSTAL_INTERACTION"] = df["RISK_INDEX"] * df["POSTAL_CODE_OTHER"]

    df = df.drop(columns=["CREDIT_SCORE", "CHILDREN", "AGE_START_DRIVING", "VEHICLE_TYPE_sports car", "AGE"],
                 errors='ignore')

    return df


def optimize_logistic_model(df):
    # Define independent (X) and dependent (y) variables
    X = df.drop(columns=["OUTCOME"])
    y = df["OUTCOME"]

    # Add constant for intercept
    X = sm.add_constant(X)

    # Fit initial logistic regression model
    logit_model = sm.Logit(y, X).fit()

    # Compute pseudo R² (McFadden's R²)
    pseudo_r2 = 1 - (logit_model.llf / logit_model.llnull)

    print("=" * 50)
    print("OPTIMIZED LOGISTIC REGRESSION MODEL SUMMARY")
    print("=" * 50)
    print(logit_model.summary())
    print(f"Pseudo R² (McFadden's R²): {pseudo_r2:.4f}")
    print(f"AIC: {logit_model.aic:.4f}")
    print(f"BIC: {logit_model.bic:.4f}")

    return X, y, logit_model


def compare_models(full_model, reduced_model):
    # Compute Pseudo R²
    full_pseudo_r2 = 1 - (full_model.llf / full_model.llnull)
    reduced_pseudo_r2 = 1 - (reduced_model.llf / reduced_model.llnull)

    # Likelihood Ratio Test
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)

    # Create DataFrame
    comparison_df = pd.DataFrame({
        "Metric": ["BIC", "AIC", "Pseudo R²", "Log-Likelihood Ratio", "Degrees of Freedom", "p-value"],
        "Full Model": [full_model.bic, full_model.aic, full_pseudo_r2, "-", "-", "-"],
        "Reduced Model": [reduced_model.bic, reduced_model.aic, reduced_pseudo_r2, lr_stat, df_diff, p_value]
    })

    print(comparison_df)


def plots_logistic_model(model, X, y, df):
    y_pred_prob = model.predict(X)  # חיזוי הסתברויות
    fpr, tpr, _ = roc_curve(y, y_pred_prob)  # חישוב ה-ROC
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='dashed', lw=1)  # קו ניחוש אקראי
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression')
    plt.legend(loc="lower right")
    plt.show()

    y_pred_prob = model.predict(X)
    precision, recall, _ = precision_recall_curve(y, y_pred_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.show()

    df["PRED_PROB"] = model.predict(X)  # הוספת עמודת הסתברות חיזוי

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df["DRIVING_EXPERIENCE"], y=df["PRED_PROB"], alpha=0.5)
    sns.lineplot(x=df["DRIVING_EXPERIENCE"], y=df["PRED_PROB"].rolling(50).mean(), color='red')
    plt.xlabel("DRIVING_EXPERIENCE")
    plt.ylabel("Predicted Probability of Claim")
    plt.title("Effect of DRIVING_EXPERIENCE on Predicted Probability")
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Activte defs -------------------------------------------------------
change_driving_exprince(df)
# visualize_variable(df)
# analyze_postal_code_influence(df)
df = map_data(df)

# correletions(df)
# visualize_variable_relationships(df)
# check_multicollinearity_and_relationships(df)

# full_linear_model(df)
# linear_model = feature_selection_linear_model(df)
# plot_fitted_vs_actual(df, linear_model)
# check_regression_assumptions(df, linear_model)

full_logistic_model = logistic_regression_analysis(df)
# SS_logit_model = stepwise_selection_logistic_regression(df)
df1 = prepare_data1_logistic_regression(df)
X, y, model1 = optimize_logistic_model(df1)
compare_models(full_logistic_model, model1)
plots_logistic_model(model1, X, y, df1)


