"""Tab Stats — extraída de streamlit_app.py sin cambiar la lógica.

Asume recovery/workouts no-None (garantizado por all_loaded()+st.stop en el
monolito). No devuelve nada: ninguna de sus variables se consume en la tab Models,
que además re-lee recovery fresco de session_state.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import streamlit as st

from app.helpers.stats import (
    compute_ecdf,
    fit_distribution,
    normality_test,
    outlier_dectection_iqr,
    outlier_detection_zscore_modified,
)


def render(recovery, workouts):
    # Data agg necessary for hypothesis testing
    recovery["Date"] = pd.to_datetime(recovery["Date"], errors="coerce")  # Convert to datetime
    workouts_daily = workouts.groupby("Date", as_index=False).agg(
        {
            "DURATION_MIN": "max",
            "WEIGHT_LBS": "max",
            "VOLUME": "sum",
            "RPE": "mean",
            "est_1RM": "mean",
        }
    )
    recovery_exercise = pd.merge(recovery, workouts_daily, on="Date", how="left").sort_values(
        "Date"
    )
    recovery_exercise["Exercise_Done"] = recovery_exercise["VOLUME"].fillna(0).gt(0).astype(int)
    recovery_exercise_done = recovery_exercise[recovery_exercise["Exercise_Done"] == 1].copy()
    recovery_exercise_notdone = recovery_exercise[recovery_exercise["Exercise_Done"] == 0].copy()
    # -----------------------------
    st.header("📉 Stats")
    st.subheader("📊 Recovery on Exercise vs Non-Exercise Days")
    allowed = sorted(
        [
            "InBed hrs",
            "Asleep hrs",
            "Wake Count",
            "REM hrs",
            "Light hrs",
            "Deep hrs",
            "Fall Asleep",
            "Overnight HRV",
            "Stress",
            "RHR",
            "Score",
        ]
    )
    check_metric = st.selectbox("Select metric to analyze:", allowed, index=allowed.index("Score"))
    picked_col = (
        recovery[check_metric]
        if recovery is not None and check_metric in recovery.columns
        else None
    )
    mean_val = picked_col.mean()
    median_val = picked_col.median()
    std_val = picked_col.std()
    trim_mean_val = stats.trim_mean(picked_col.dropna(), 0.1) if picked_col is not None else None
    n = picked_col.dropna().shape[0]
    trim_mean = stats.trim_mean(picked_col.dropna(), 0.1) if picked_col is not None else None
    cv = std_val / mean_val if mean_val not in (0, None) and not pd.isna(mean_val) else np.nan
    col_pvalue, col_inter = normality_test(picked_col) if picked_col is not None else (None, None)
    # ------------------------------------- 4 MOMENTS OF DATA ----------------------------------------
    with st.expander("🎯 Four Moments of Statistics", expanded=False):
        st.subheader("🎯 Four Moments of Statistics")
        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
        series_for_chart = (
            picked_col.dropna().astype(float).tail(30).tolist() if picked_col is not None else []
        )
        c1.metric("Median", f"{median_val:.2f}" if median_val is not None else "—")
        c2.metric(
            "Mean",
            f"{mean_val:.2f}" if mean_val is not None and not pd.isna(mean_val) else "—",
            chart_data=series_for_chart if len(series_for_chart) >= 2 else None,
            chart_type="line",
        )
        c3.metric("Std Dev", f"{std_val:.2f}" if std_val is not None else "—")
        c4.metric("Trimmed Mean (10%)", f"{trim_mean:.2f}" if trim_mean is not None else "—")
        c5.metric(
            "Sample (n)",
            "Sufficient" if n >= 30 else "Insufficient",
            delta=n,
            delta_color="normal" if n >= 30 else "inverse",
            help="n >=30 is considered sufficient for Central Limit Theorem.",
            delta_arrow="off",
        )
        c6.metric(
            "Coef of Var (CV)",
            "Good"
            if cv is not None and cv < 0.1
            else "Acceptable"
            if cv is not None and cv < 0.2
            else "High",
            delta=f"{round(cv * 100, 2)} %",
            delta_color="normal"
            if cv is not None and cv < 0.1
            else "orange"
            if cv is not None and cv < 0.2
            else "inverse"
            if cv is not None
            else None,
            help="CV <10% is considered good stability; 10-20% acceptable; >20% high variability.",
        )
        c7.metric(
            "Skewness",
            f"{picked_col.skew():.2f}" if picked_col is not None else "—",
            help="Skewness indicates asymmetry. >0 means right-skewed, <0 means left-skewed.",
        )
        c8.metric(
            "Kurtosis",
            f"{picked_col.kurtosis():.2f}" if picked_col is not None else "—",
            help="Kurtosis indicates the 'tailedness' of the distribution. >3 means heavy tails, <3 means light tails.",
            delta="Leptokurtic"
            if picked_col is not None and picked_col.kurtosis() > 3
            else "Platykurtic"
            if picked_col is not None and picked_col.kurtosis() < 3
            else "Mesokurtic"
            if picked_col is not None
            else None,
            delta_arrow="off",
            delta_color="green"
            if picked_col is not None and picked_col.kurtosis() == 3
            else "red"
            if picked_col is not None and picked_col.kurtosis() > 3
            else "green"
            if picked_col is not None and picked_col.kurtosis() < 3
            else None,
        )
    # ------------------------------------- EMPIRICAL CDF & PERCENTILES ----------------------------------------
    with st.expander("📊 Empirical CDF & Percentiles", expanded=False):
        st.subheader("📊 Empirical CDF & Percentiles")
        c1, c2 = st.columns(2)
        with c1:
            if picked_col is not None and not picked_col.dropna().empty:
                complementary = st.segmented_control(
                    "Complementary CDF ?:",
                    [True, False],
                    key="cdf_type_control",
                    default=True,
                    help="If True then complementary CDF (1 - CDF) is shown.",
                )
                perc_90 = picked_col.quantile(0.9)
                perc_75 = picked_col.quantile(0.75)
                fig, ax = plt.subplots(figsize=(10, 4))
                cecdf_50 = compute_ecdf(
                    picked_col.dropna(), median_val, complementary=complementary
                )
                cecdf_75 = compute_ecdf(picked_col.dropna(), perc_75, complementary=complementary)
                cecdf_90 = compute_ecdf(picked_col.dropna(), perc_90, complementary=complementary)
                sns.ecdfplot(
                    data=recovery,
                    x=check_metric,
                    label=f"Empirical CDF {check_metric}",
                    color="green",
                    complementary=complementary,
                    ax=ax,
                )
                plt.axvline(
                    mean_val,
                    color="blue",
                    linestyle="--",
                    label=f"Mean: {mean_val:.2f}",
                    linewidth=0.5,
                )
                plt.axvline(
                    median_val,
                    color="lightseagreen",
                    linestyle=":",
                    label=f"50th Percentile: {median_val:.2f}",
                    linewidth=1,
                )
                plt.axvline(
                    perc_90,
                    color="yellow",
                    linestyle="--",
                    label=f"90th Percentile: {perc_90:.2f}",
                    linewidth=0.5,
                )
                plt.axvline(
                    perc_75,
                    color="brown",
                    linestyle="--",
                    label=f"75th Percentile: {perc_75:.2f}",
                    linewidth=0.5,
                )
                if complementary:
                    plt.title(
                        f"Complementary ECDF of {check_metric}",
                        fontsize=14,
                        fontweight="bold",
                        pad=15,
                    )
                    plt.ylabel("Complementary ECDF")
                else:
                    plt.title(
                        f"Empirical CDF of {check_metric}", fontsize=14, fontweight="bold", pad=15
                    )
                    plt.ylabel("ECDF")
                plt.xlabel(check_metric)
                plt.legend(loc="best", fontsize=7)
                sns.despine()
                st.pyplot(fig)
            else:
                st.warning(f"No data available for {check_metric} to plot ECDF.")
        with c2:
            st.subheader(f"📈 Percentile Insights {check_metric}")
            st.write("Typical value (median)", round(median_val, 2))
            st.write("Uncommon value (75th ):", round(perc_75, 2))
            st.write("Rare high value (90th):", round(perc_90, 2))
            st.subheader(f"📊 Probability Insights {check_metric}")
            if complementary:
                st.write(
                    "The probability of exceeding",
                    round(median_val, 2),
                    " is:",
                    round((cecdf_50) * 100, 2),
                    " %",
                )
                st.write(
                    "The probability of exceeding",
                    round(perc_75, 2),
                    " is:",
                    round((cecdf_75) * 100, 2),
                    " %",
                )
                st.write(
                    "The probability of exceeding",
                    round(perc_90, 2),
                    " is:",
                    round((cecdf_90) * 100, 2),
                    " %",
                )
            else:
                st.write(
                    "The probability of getting any value up to my typical performance is:",
                    round(cecdf_50 * 100, 2),
                    " %",
                )
                st.write(
                    "The probability of getting any value up to common performance is:",
                    round(cecdf_75 * 100, 2),
                    " %",
                )
                st.write(
                    "The probability of getting any value up to atypical performance is:",
                    round(cecdf_90 * 100, 2),
                    " %",
                )
            st.info(
                "Note: Even though the ECDF provides empirical probabilities based on historical data, "
                "it does not guarantee future outcomes. Use this information as a guide rather than a definitive prediction.",
                icon="ℹ️",
            )
    # ------------------------------------- NORMALITY TEST & VISUALS ----------------------------------------
    with st.expander("🔍 Normality Test for Recovery", expanded=False):
        st.subheader("🔍 Normality Test for Recovery")

        # Interpretation
        if col_pvalue is not None and col_pvalue > 0.05:
            st.success(
                f"Shapiro Wilk Test: {check_metric} appears to be normally distributed (p={col_pvalue:.3f}). You can use parametric tests."
            )
        elif col_pvalue is not None and col_pvalue <= 0.05:
            st.info(
                f"Shapiro Wilk Test: {check_metric} does not appear to be normally distributed (p={col_pvalue:.3f}). You may want to use non-parametric tests."
            )
        else:
            st.info(f"Not enough data to perform normality test on {check_metric}.")
        try:
            distributions = pd.to_numeric(picked_col, errors="coerce").dropna()
            st.dataframe(fit_distribution(distributions))
        except Exception as e:
            st.error(f"Error fitting distribution: {e}")
        bins = st.slider(
            "Select number of bins for histogram", 5, 50, 20, 1, width=250, key="hist_bins_slider"
        )
        c1, c2 = st.columns(2)
        with c1:
            # Plot histogram
            fig, ax = plt.subplots(figsize=(7, 3))
            sns.histplot(picked_col.dropna(), kde=True, ax=ax, stat="probability", bins=bins)
            ax.axvline(mean_val, color="blue", linestyle="--", label=f"Mean: {mean_val:.2f}")
            ax.axvline(median_val, color="red", linestyle=":", label=f"Median: {median_val:.2f}")
            ax.axvline(
                trim_mean_val,
                color="green",
                linestyle="-.",
                label=f"Trimmed Mean: {trim_mean_val:.2f}",
            )
            ax.axvspan(
                mean_val - std_val,
                mean_val + std_val,
                color="yellow",
                alpha=0.15,
                label=f"±1 Std Dev: {std_val:.2f}",
            )
            ax.set_title(f"Histogram of {check_metric}", fontsize=14, fontweight="bold", pad=15)
            ax.legend(loc="best", fontsize=7)
            ax.set_xlabel(check_metric)
            ax.set_ylabel("Probability")
            sns.despine(ax=ax)
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(7, 3))
            sns.boxplot(
                y=picked_col.dropna(),
                ax=ax,
                width=0.3,
                fliersize=3,
                flierprops={"markerfacecolor": "red", "marker": "o"},
            )
            sns.stripplot(
                y=picked_col.dropna(), ax=ax, color="lightblue", size=4, jitter=True, alpha=0.15
            )
            ax.set_title(f"Boxplot of {check_metric}", fontsize=14, fontweight="bold", pad=15)
            ax.set_xlabel(check_metric)
            sns.despine(ax=ax)
            st.pyplot(fig)

        if st.checkbox("Show full time series plot", value=True, key="full_time_series_checkbox"):
            fig, ax = plt.subplots(figsize=(10, 3))
            sns.lineplot(data=recovery, x="Date", y=check_metric, ax=ax, linewidth=1)
            ax.axhline(mean_val, color="blue", linestyle="--", label=f"Mean: {mean_val:.2f}")
            ax.axhline(median_val, color="red", linestyle=":", label=f"Median: {median_val:.2f}")
            ax.axhline(
                trim_mean_val,
                color="green",
                linestyle="-.",
                label=f"Trimmed Mean: {trim_mean_val:.2f}",
            )
            ax.axhspan(
                mean_val - std_val,
                mean_val + std_val,
                color="yellow",
                alpha=0.05,
                label=f"±1 Std Dev: {std_val:.2f}",
            )
            ax.legend(loc="best", fontsize=5)
            ax.set_title(f"Time Series of {check_metric}", fontsize=14, fontweight="bold", pad=15)
            ax.set_xlabel("")
            ax.set_ylabel(check_metric)
            sns.despine(ax=ax)
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)
        else:
            date_plot = st.slider(
                "Select number of days to show for time series plot",
                30,
                365,
                180,
                1,
                key="time_series_days_slider",
                width=250,
            )
            date_filter = recovery["Date"].max() - pd.Timedelta(days=date_plot)
            fig, ax = plt.subplots(figsize=(10, 3))
            sns.lineplot(
                data=recovery[recovery["Date"] >= date_filter],
                x="Date",
                y=check_metric,
                ax=ax,
                linewidth=1,
            )
            ax.axhline(mean_val, color="blue", linestyle="--", label=f"Mean: {mean_val:.2f}")
            ax.axhline(median_val, color="red", linestyle=":", label=f"Median: {median_val:.2f}")
            ax.axhline(
                trim_mean_val,
                color="green",
                linestyle="-.",
                label=f"Trimmed Mean: {trim_mean_val:.2f}",
            )
            ax.axhspan(
                mean_val - std_val,
                mean_val + std_val,
                color="yellow",
                alpha=0.05,
                label=f"±1 Std Dev: {std_val:.2f}",
            )
            ax.legend(loc="best", fontsize=5)
            ax.set_title(f"Time Series of {check_metric}", fontsize=14, fontweight="bold", pad=15)
            ax.set_xlabel("")
            ax.set_ylabel(check_metric)
            sns.despine(ax=ax)
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)
    # ------------------------------------- OUTLIERS DETECTION ----------------------------------------
    with st.expander("🧪 Outliers Detection", expanded=False):
        st.subheader("🧪 Outliers Detection")
        outliers_iqr = (
            outlier_dectection_iqr(picked_col) if picked_col is not None else pd.Series(dtype=float)
        )
        outliers_z = outlier_detection_zscore_modified(picked_col, threshold=3)

        if len(outliers_iqr) == 0 and len(outliers_z) == 0:
            st.success(
                f"No outliers detected in {check_metric} using IQR method and Modified Z-Score method.",
                icon="✅",
            )
        else:
            st.info(
                f"Detected {len(outliers_iqr)} outlier(s) in {check_metric} using IQR method.",
                icon="ℹ️",
            )
            st.dataframe(outliers_iqr.to_frame(name=f"{check_metric} Value"))
            st.info(
                f"Detected {len(outliers_z)} outlier(s) in {check_metric} using Modified Z-Score method.",
                icon="🚨",
            )
            st.dataframe(outliers_z.to_frame(name=f"{check_metric} Value"))

    # ------------------------------------- HYPOTHESIS TESTING ----------------------------------------
    with st.expander("🛠️ Tests with rest days and exercise days", expanded=False):
        st.subheader("🛠️ Statistical Tests")
        if col_pvalue > 0.05:
            group1 = recovery_exercise_done[check_metric].dropna()
            group2 = recovery_exercise_notdone[check_metric].dropna()
            st.write(
                "Since the data appears to be normally distributed, you can use parametric tests such as t-tests or ANOVA for further analysis."
            )
            options = ["One-sample t-test", "Independent t-test"]
            choice = st.segmented_control(
                "Select test to perform:", options=options, key="parametric_tests_control"
            )
            if choice == "One-sample t-test":
                c1, c2 = st.columns(2)
                with c1:
                    option_df = st.selectbox(
                        "Select data group to test against population mean:",
                        ["Exercise Days", "Rest Days", "All Days"],
                    )
                    if option_df == "Exercise Days":
                        group = recovery_exercise_done[check_metric].dropna()
                        st.warning(
                            "T Test are about means. They test whether the mean is different from the population mean (One-sample test) or whether the means of the groups are significantly different (Independent test).",
                            icon="⚠️",
                        )
                        popmean = st.number_input(
                            "Enter population mean to compare against:",
                            value=float(mean_val)
                            if mean_val is not None and not pd.isna(mean_val)
                            else 0.0,
                        )
                        alternative = st.selectbox(
                            "Select alternative hypothesis:", ["two-sided", "less", "greater"]
                        )
                        ttest_res = stats.ttest_1samp(group, popmean, alternative=alternative)
                        button_run = st.button("Run One-sample t-test")
                        if button_run:
                            st.write(
                                f"t-statistic: {ttest_res.statistic:.3f},"
                                f" p-value: {ttest_res.pvalue:.3f}"
                            )
                            if ttest_res.pvalue < 0.05:
                                st.success("Reject the null hypothesis at α=0.05 level.")
                            else:
                                st.info("Fail to reject the null hypothesis at α=0.05 level.")
                    elif option_df == "Rest Days":
                        group = recovery_exercise_notdone[check_metric].dropna()
                        st.warning(
                            "T Test are about means. They test whether the mean is different from the population mean (One-sample test) or whether the means of the groups are significantly different (Independent test).",
                            icon="⚠️",
                        )
                        popmean = st.number_input(
                            "Enter population mean to compare against:",
                            value=float(mean_val)
                            if mean_val is not None and not pd.isna(mean_val)
                            else 0.0,
                        )
                        alternative = st.selectbox(
                            "Select alternative hypothesis:", ["two-sided", "less", "greater"]
                        )
                        ttest_res = stats.ttest_1samp(group, popmean, alternative=alternative)
                        button_run = st.button("Run One-sample t-test")
                        if button_run:
                            st.write(
                                f"t-statistic: {ttest_res.statistic:.3f},"
                                f" p-value: {ttest_res.pvalue:.3f}"
                            )
                            if ttest_res.pvalue < 0.05:
                                st.success("Reject the null hypothesis at α=0.05 level.")
                            else:
                                st.info("Fail to reject the null hypothesis at α=0.05 level.")
                    elif option_df == "All Days":
                        group = recovery[check_metric].dropna()
                        st.warning(
                            "T Test are about means. They test whether the mean is different from the population mean (One-sample test) or whether the means of the groups are significantly different (Independent test).",
                            icon="⚠️",
                        )
                        popmean = st.number_input(
                            "Enter population mean to compare against:",
                            value=float(mean_val)
                            if mean_val is not None and not pd.isna(mean_val)
                            else 0.0,
                        )
                        alternative = st.selectbox(
                            "Select alternative hypothesis:", ["two-sided", "less", "greater"]
                        )
                        ttest_res = stats.ttest_1samp(group, popmean, alternative=alternative)
                        button_run = st.button(
                            "Run One-sample t-test", key="one_sample_ttest_button"
                        )
                        if button_run:
                            st.write(
                                f"t-statistic: {ttest_res.statistic:.3f},"
                                f" p-value: {ttest_res.pvalue:.3f}"
                            )
                            if ttest_res.pvalue < 0.05:
                                st.success("Reject the null hypothesis at α=0.05 level.")
                            else:
                                st.info("Fail to reject the null hypothesis at α=0.05 level.")
                with c2:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    sns.kdeplot(group, color="lightblue", label=f"{option_df}", ax=ax)
                    ax.axvline(
                        group.mean(),
                        color="blue",
                        linestyle="--",
                        label=f"Exercise Mean: {group.mean():.2f}",
                    )
                    ax.axvline(
                        popmean,
                        color="red",
                        linestyle="--",
                        label=f"Population Mean: {popmean:.2f}",
                    )
                    sns.despine(ax=ax)
                    plt.title(f"Distribution of {check_metric}")
                    plt.xlabel(check_metric)
                    plt.ylabel("Density")
                    plt.legend(loc="best", fontsize=7)
                    st.pyplot(fig)

            elif choice == "Independent t-test":
                c1, c2 = st.columns(2)
                with c1:
                    st.warning(
                        "T Test are about means. They test whether the mean is different from the population mean (One-sample test) or whether the means of the groups are significantly different (Independent test).",
                        icon="⚠️",
                    )
                    st.info(f"Group 1 is {check_metric} with exercise ({group1.shape[0]} samples)")
                    st.info(f"Group 2 is {check_metric} on rest days {group2.shape[0]} samples)")
                    st.write("Exercise mean:", float(group1.mean()))
                    st.write("Rest mean:", float(group2.mean()))
                    st.write("Δ mean (ex - rest):", float(group1.mean() - group2.mean()))
                    alternative = st.selectbox(
                        "Select alternative hypothesis:", ["two-sided", "less", "greater"]
                    )
                    ttest2_res = stats.ttest_ind(
                        group1, group2, alternative=alternative, equal_var=False
                    )
                    button_run2 = st.button(
                        "Run Independent t-test", key="independent_ttest_button"
                    )
                    if button_run2:
                        st.write(
                            f"t-statistic: {ttest2_res.statistic:.3f},"
                            f" p-value: {ttest2_res.pvalue:.3f}"
                        )
                        if ttest2_res.pvalue < 0.05:
                            st.success("Reject the null hypothesis at α=0.05 level.")
                        else:
                            st.info("Fail to reject the null hypothesis at α=0.05 level.")
                with c2:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    sns.kdeplot(group1, color="lightblue", label="Exercise Days", ax=ax)
                    sns.kdeplot(group2, color="salmon", label="Rest Days", ax=ax)
                    ax.axvline(
                        group1.mean(),
                        color="blue",
                        linestyle="--",
                        label=f"Exercise Mean: {group1.mean():.2f}",
                    )
                    ax.axvline(
                        group2.mean(),
                        color="red",
                        linestyle="--",
                        label=f"Rest Mean: {group2.mean():.2f}",
                    )
                    sns.despine(ax=ax)
                    plt.title(f"Distribution of {check_metric}")
                    plt.xlabel(check_metric)
                    plt.ylabel("Density")
                    plt.legend(loc="best", fontsize=7)
                    st.pyplot(fig)
        elif col_pvalue <= 0.05:
            st.write(
                "Since the data does not appear to be normally distributed, you can use non-parametric tests such as the Wilcoxon signed-rank test or the Mann-Whitney U test for further analysis."
            )
            options = ["Spearman Correlation", "Mann-Whitney U test"]
            choice = st.segmented_control("Select test to perform:", options=options)
            if choice == "Spearman Correlation":
                col2 = st.selectbox(
                    "Select another metric to correlate with:",
                    allowed,
                    index=allowed.index("Stress"),
                )
                # Align pairs by dropping rows where either metric is NaN
                df_pair = recovery[[check_metric, col2]].dropna()
                st.info(f"Using {df_pair.shape[0]} paired observations for correlation.")
                npairs = df_pair.shape[0]
                if npairs < 500:
                    st.warning(
                        "Spearman correlation is accurate for large samples (over 500 samples). For smaller samples, interpret results with caution.",
                        icon="⚠️",
                    )
                if df_pair.shape[0] < 2:
                    st.warning("Need at least 2 paired observations to compute correlation.")
                else:
                    x = df_pair[check_metric].astype(float)
                    y = df_pair[col2].astype(float)
                    try:
                        alternative = st.selectbox(
                            "Select alternative hypothesis for Spearman correlation:",
                            ["two-sided", "less", "greater"],
                            key="spearman_alternative_selectbox",
                        )
                        spearman_corr = stats.spearmanr(x, y, alternative=alternative)
                    except Exception as e:
                        st.error(f"Could not compute Spearman correlation: {e}")
                    else:
                        button_run = st.button(
                            "Run Spearman Correlation", key="spearman_corr_button"
                        )
                        if button_run:
                            c1, c2 = st.columns(2)
                            with c1:
                                coef_raw = spearman_corr.statistic
                                p_raw = spearman_corr.pvalue

                                coef_spearman = (
                                    coef_raw[0, 1]
                                    if hasattr(coef_raw, "ndim") and coef_raw.ndim > 0
                                    else coef_raw
                                )
                                p_value_val = (
                                    p_raw[0, 1]
                                    if hasattr(p_raw, "ndim") and p_raw.ndim > 0
                                    else p_raw
                                )

                                st.write(f"Spearman correlation coefficient: {coef_spearman}")
                                st.write(f"p-value: {p_value_val}")

                                if p_value_val < 0.05:
                                    st.success("Reject the null hypothesis at α=0.05 level.")
                                else:
                                    st.info("Fail to reject the null hypothesis at α=0.05 level.")
                            with c2:
                                fig, ax = plt.subplots(figsize=(7, 5))
                                sns.scatterplot(x=x, y=y, ax=ax, alpha=0.7)
                                sns.regplot(
                                    x=x, y=y, lowess=True, scatter=False, ax=ax, color="orange"
                                )
                                ax.set_title(
                                    f"Spearman correlation (Spearman coef = {coef_spearman:.2f}, p = {p_value_val:.3f})",
                                    fontsize=14,
                                    fontweight="bold",
                                    pad=15,
                                )
                                ax.set_xlabel(check_metric)
                                ax.set_ylabel(col2)
                                sns.despine(ax=ax)
                                st.pyplot(fig)

            elif choice == "Mann-Whitney U test":
                group1 = recovery_exercise_done[check_metric].dropna()
                group2 = recovery_exercise_notdone[check_metric].dropna()
                c1, c2 = st.columns(2)
                with c1:
                    st.warning(
                        "Mann-Whitney U is about distributions, not means. It tests whether values from one group tend to be higher than the other.",
                        icon="⚠️",
                    )
                    st.info(f"Group 1 is {check_metric} with exercise ({group1.shape[0]} samples)")
                    st.info(f"Group 2 is {check_metric} on rest days ({group2.shape[0]} samples)")
                    st.write("Exercise median:", float(group1.median()))
                    st.write("Rest median:", float(group2.median()))
                    st.write("Δ median (ex - rest):", float(group1.median() - group2.median()))
                    alternative = st.selectbox(
                        "Select alternative hypothesis:", ["two-sided", "less", "greater"]
                    )
                    button_run2 = st.button("Run Mann-Whitney U test", key="mwu_test_button")
                    if button_run2:
                        stats_mwu, pvalue_mwu = stats.mannwhitneyu(
                            group1, group2, alternative=alternative
                        )
                        st.write("U statistic: ", stats_mwu)
                        st.write("p-value: ", pvalue_mwu)
                        if pvalue_mwu < 0.05:
                            st.success("Reject the null hypothesis at α=0.05 level.")
                        else:
                            st.info("Fail to reject the null hypothesis at α=0.05 level.")
                        u = stats_mwu
                        n1 = len(group1)
                        n2 = len(group2)
                        # Calculate effect size
                        CLES = u / (n1 * n2)
                        st.write("Common Language Effect Size (CLES):", round(CLES, 2))
                        if CLES > 0.5:
                            st.write(
                                f"{CLES * 100:.1f}% chance that a randomly selected exercise day has a higher "
                                f"{check_metric} than a randomly selected rest day."
                            )
                        elif CLES < 0.5:
                            st.write(
                                f"{(1 - CLES) * 100:.1f}% chance that a randomly selected rest day has a higher "
                                f"{check_metric} than a randomly selected exercise day."
                            )
                        else:
                            st.write(
                                "No difference between exercise and rest days in "
                                + check_metric
                                + "."
                            )
                with c2:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    sns.kdeplot(group1, color="lightblue", label="Exercise Days", ax=ax)
                    sns.kdeplot(group2, color="salmon", label="Rest Days", ax=ax)
                    sns.despine(ax=ax)
                    plt.title(
                        f"Distribution of {check_metric}", fontsize=14, fontweight="bold", pad=15
                    )
                    plt.xlabel(check_metric)
                    plt.ylabel("Density")
                    plt.legend(loc="best")
                    st.pyplot(fig)
