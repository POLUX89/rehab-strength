"""Tab Home (Weekly Snapshot) — extraída de streamlit_app.py sin cambiar la lógica.

render(workouts, sleep, recovery) -> sleep. Devuelve sleep porque la tab lo
reasigna con sort_values (copia), igual que Recovery/Sleep; el monolito hace
sleep = home_tab.render(...) para conservar esa copia en la variable global.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from app.helpers.transforms import pick_col, recovery_zone, safe_minimal_last, week_bounds


def render(workouts, sleep, recovery):
    st.header("🏠 Weekly Snapshot")
    start_wk, end_wk = week_bounds()
    st.caption(f"Week: {start_wk.date()} → {end_wk.date()}")
    st.markdown("---")
    # compute last dates safely
    st.subheader("🔐 Data Integrity")
    last_workouts = workouts["Date"].max() if "Date" in workouts.columns else workouts["DATE"].max()
    last_sleep = sleep["Date"].max() if sleep is not None and "Date" in sleep.columns else None
    last_recovery = (
        recovery["Date"].max() if recovery is not None and "Date" in recovery.columns else None
    )

    # freshness (days old)
    today = pd.Timestamp.today().normalize()

    def age_days(ts):
        if ts is None or pd.isna(ts):
            return None
        return int((today - pd.to_datetime(ts).normalize()).days)

    a_w = age_days(last_workouts)
    a_s = age_days(last_sleep)
    a_r = age_days(last_recovery)

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 2.4])
    c1.metric(
        "Workouts last",
        f"{pd.to_datetime(last_workouts):%b %d}",
        f"{a_w}d old" if a_w is not None else "—",
        delta_arrow="off",
    )
    c2.metric(
        "Sleep last",
        f"{pd.to_datetime(last_sleep):%b %d}" if last_sleep is not None else "—",
        f"{a_s}d old" if a_s is not None else "—",
        delta_arrow="off",
    )
    c3.metric(
        "Recovery last",
        f"{pd.to_datetime(last_recovery):%b %d}" if last_recovery is not None else "—",
        f"{a_r}d old" if a_r is not None else "—",
        delta_arrow="off",
    )

    # optional: overall status label
    overall_age = max([x for x in [a_w, a_s, a_r] if x is not None], default=None)
    label = (
        "🟢 Fresh"
        if (overall_age is not None and overall_age <= 1)
        else (
            "🟡 Slightly delayed"
            if (overall_age is not None and overall_age <= 3)
            else "🔴 Outdated"
        )
    )
    c4.markdown(f"**Data status:** {label}")
    st.markdown("---")
    # -------------------------
    # Weekly workouts snapshot
    # -------------------------
    if workouts is None:
        st.info("Upload workouts CSV to see weekly snapshot.")
    else:
        if "DAY" not in workouts.columns or workouts["DAY"].isna().all():
            st.warning("Workouts CSV needs a DATE/Date column so the app can create 'DAY'.")
        else:
            wk = workouts[(workouts["DAY"] >= start_wk) & (workouts["DAY"] <= end_wk)].copy()

            # Workouts (sessions): unique (DAY, WORKOUT_NAME) if available
            if "WORKOUT_NAME" in wk.columns:
                workouts_count = (
                    wk.dropna(subset=["WORKOUT_NAME"]).groupby(["DAY", "WORKOUT_NAME"]).ngroups
                )
            else:
                workouts_count = wk["DAY"].nunique()

            # Time exercised: MAX duration per session, then sum
            total_minutes = None
            if "DURATION_MIN" in wk.columns:
                wk["DURATION_MIN"] = pd.to_numeric(wk["DURATION_MIN"], errors="coerce")

                if "WORKOUT_NAME" in wk.columns:
                    total_minutes = (
                        wk.dropna(subset=["DURATION_MIN"])
                        .groupby(["DAY", "WORKOUT_NAME"])["DURATION_MIN"]
                        .max()
                        .sum()
                    )
                else:
                    total_minutes = wk.groupby("DAY")["DURATION_MIN"].max().sum()

            total_hours = None if total_minutes is None else float(total_minutes) / 60.0

            # -------------------------
            # Latest values (sleep/recovery)
            # -------------------------
            st.subheader("📊 Key Metrics")
            last_sigmoid = (
                safe_minimal_last(recovery, "Date", "Sigmoid Recovery Score")
                if recovery is not None
                else None
            )
            sleep_score_col = pick_col(recovery, ["Score", "Sleep Score", "SleepScore", "SCORE"])
            sleep_hrv_col = pick_col(recovery, ["Overnight HRV", "Avg. HRV", "HRV", "7d Avg"])

            last_sleep_score = safe_minimal_last(recovery, "Date", sleep_score_col)
            last_hrv = safe_minimal_last(recovery, "Date", sleep_hrv_col)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Workouts (week)", int(workouts_count) if workouts_count is not None else "—")
            c2.metric(
                "Time exercised (hrs)",
                f"{total_hours:.1f}" if total_hours is not None else "—",
                delta="4 Hrs goal",
                delta_arrow="off",
                delta_color="normal" if total_hours is not None and total_hours >= 4 else "inverse",
            )
            if last_sigmoid is None or pd.isna(last_sigmoid):
                c3.metric("Last Recovery", "—", "No data")
            else:
                c3.metric(
                    "Last Recovery",
                    f"{last_sigmoid:.3f}",
                    recovery_zone(last_sigmoid),
                    delta_arrow="off",
                    delta_color="normal"
                    if last_sigmoid is not None and last_sigmoid >= 0.7
                    else (
                        "inverse"
                        if last_sigmoid is not None and last_sigmoid >= 0.55
                        else "inverse"
                    ),
                )
            c4.metric(
                "Last sleep score %",
                f"{float(last_sleep_score):.0f}" if last_sleep_score is not None else "—",
                "Excellent"
                if last_sleep_score is not None and last_sleep_score >= 85
                else (
                    "Fair" if last_sleep_score is not None and last_sleep_score >= 70 else "Poor"
                ),
                delta_arrow="off",
            )
            c5.metric(
                "Last HRV (ms)",
                f"{float(last_hrv):.0f}"
                if last_hrv is not None and str(last_hrv) != "nan"
                else "—",
                delta="Bad"
                if last_hrv is not None and last_hrv < 45
                else "Good"
                if last_hrv is not None and last_hrv <= 60
                else "Excellent",
                delta_arrow="off",
                delta_color="normal"
                if last_hrv is not None and last_hrv >= 45
                else ("inverse" if last_hrv is not None and last_hrv < 60 else "off"),
            )

            st.subheader("📈 Recent Trends")
            last_sigmoid_nap = (
                safe_minimal_last(recovery, "Date", "Sigmoid with Nap")
                if recovery is not None
                else None
            )
            last_delta = (
                safe_minimal_last(recovery, "Date", "DELTA_NAP") if recovery is not None else None
            )
            last_nap_status = (
                safe_minimal_last(recovery, "Date", "Nap_Status") if recovery is not None else None
            )
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Recovery with Nap",
                f"{last_sigmoid_nap:.3f}" if last_sigmoid_nap is not None else "—",
                recovery_zone(last_sigmoid_nap),
                delta_arrow="off",
                delta_color="normal"
                if last_sigmoid_nap is not None and last_sigmoid_nap > last_sigmoid
                else (
                    "inverse"
                    if last_sigmoid_nap is not None and last_sigmoid_nap < last_sigmoid
                    else "off"
                ),
            )
            c2.metric(
                "Δ Nap Effect",
                f"{last_delta:.3f}" if last_delta is not None else "—",
                f"{'⬆️ Positive' if last_delta is not None and last_delta > 0 else ('⬇️ Negative' if last_delta is not None and last_delta < 0 else 'No effect')}",
                delta_arrow="off",
                delta_color="normal"
                if last_delta is not None and last_delta > 0
                else ("inverse" if last_delta is not None and last_delta < 0 else "off"),
            )
            c3.metric(
                "Last Nap Status",
                last_nap_status if last_nap_status is not None else "No nap",
                delta_arrow="off",
            )
    st.markdown("---")

    # ---------------------------
    # Naps summary
    # --------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("💤 Naps logged")
        window = st.segmented_control(
            "Select nap days window for avg:",
            options=[7, 14, 30, 60, 90],
            default=14,
            key="nap_avg_window",
            selection_mode="single",
            format_func=lambda x: f"{x} days",
        )
        if window is None:
            window = 14
            st.markdown("Please select a number of days")
        sleep = sleep.sort_values("Date", ascending=True)
        avg_nap = sleep.tail(window)["Asleep_Nap"].dropna().mean()
        try:
            if pd.isna(avg_nap):
                col1.metric(f"💤 Avg nap time (last {window} days)", " 0 minutes")
                st.write("No Naps Logged")
            else:
                col1.metric(f"💤 Avg nap time (last {window} days)", f"{avg_nap:.1f} minutes")
        except Exception as e:
            st.error(f"Error computing nap frequency: {e}")
            st.error("Please select a number")

    with col2:
        st.subheader("📅 Nap Days")
        window2 = st.segmented_control(
            "Select nap days window for avg:",
            options=[7, 14, 30, 60, 90],
            default=14,
            key="nap_avg_window2",
            selection_mode="single",
            format_func=lambda x: f"{x} days",
        )
        try:
            if pd.isna(avg_nap):
                st.write("No Day Naps Logged")
            else:
                start_date = sleep["Date"].max() - pd.Timedelta(days=window2)
                nap_data = sleep[sleep["Date"] >= start_date].dropna(subset=["Asleep_Nap"])
                n_naps = nap_data.loc[
                    nap_data["Asleep_Nap"].notna() & (nap_data["Asleep_Nap"] > 0)
                ].shape[0]
                col2.metric("Naps logged", f"{n_naps}")
        except Exception as e:
            st.error(f"Error computing nap frequency: {e}")
            st.error("Please select a number")
    with col3:
        st.subheader("📈 Nap Frequency")
        window3 = st.segmented_control(
            "Select nap days window for avg:",
            options=[7, 14, 30, 60, 90],
            default=14,
            key="nap_avg_window3",
            selection_mode="single",
            format_func=lambda x: f"{x} days",
        )
        try:
            if pd.isna(avg_nap):
                st.write("No Frequency Naps Logged")
            else:
                start_date = sleep["Date"].max() - pd.Timedelta(days=window3)
                nap_data = sleep[sleep["Date"] >= start_date].dropna(subset=["Asleep_Nap"])
                sleep_filtered = sleep[sleep["Date"] >= start_date]
                freq_val = (
                    nap_data[nap_data["Asleep_Nap"].notna() & (nap_data["Asleep_Nap"] > 0)].shape[0]
                    / sleep_filtered.shape[0]
                ) * 100

                def freq_label(x):
                    if pd.isna(x):
                        return "No data"
                    if x <= 15:
                        return "🔴 Low"
                    if x <= 30:
                        return "🟡 Moderate"
                    return "🟢 High"

                col3.metric("Nap frequency", f"{freq_val:.1f} %", delta=freq_label(freq_val))
                st.caption(
                    f"Naps on {nap_data[nap_data['Asleep_Nap'].notna() & (nap_data['Asleep_Nap'] > 0)].shape[0]} of the last {window3} days"
                )
        except Exception as e:
            st.error(f"Error computing nap frequency: {e}")
            st.error("Please select a number")
    st.markdown("---")
    # -------------------------
    # Quick trends (independent)
    # -------------------------
    left, right = st.columns(2)

    with left:
        st.subheader("🧠 Recovery (last 14 days)")
        if recovery is not None and {"Date", "Sigmoid Recovery Score"}.issubset(recovery.columns):
            tmp = (
                recovery.dropna(subset=["Date", "Sigmoid Recovery Score"])
                .sort_values("Date")
                .tail(14)
            )
            if tmp.empty:
                st.info("Recovery CSV loaded but no usable rows.")
            else:
                fig, ax = plt.subplots(figsize=(7, 3))
                tmp_avg = tmp["Sigmoid Recovery Score"].mean()
                sld1 = st.slider("Select days for moving average", 2, 11, 5, 1, width=250)
                ax.plot(
                    tmp["Date"],
                    tmp["Sigmoid Recovery Score"],
                    marker="o",
                    markersize=3,
                    color="green",
                )
                roll_avg_recovery = tmp["Sigmoid Recovery Score"].rolling(window=sld1).mean()
                ax.plot(
                    tmp["Date"],
                    roll_avg_recovery,
                    color="orange",
                    linestyle="--",
                    alpha=0.25,
                    label=f"MA {sld1} days {roll_avg_recovery.iloc[-1]:.2f}",
                )
                ax.axhline(
                    tmp_avg, color="blue", linestyle=":", alpha=0.6, label=f"Avg {tmp_avg:.2f}"
                )
                ax.set_xlabel("")
                ax.set_ylim(0, 1)
                ax.legend(loc="lower left")
                ax.tick_params(axis="x", rotation=45, labelsize=6)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                ax.set_ylabel("Sigmoid")
                sns.despine(ax=ax)
                st.pyplot(fig)
        else:
            st.info("No recovery data uploaded yet.")

    with right:
        st.subheader("🧠 Recovery (last 14 days) with Nap")
        if recovery is not None and {"Date", "Sigmoid with Nap"}.issubset(recovery.columns):
            tmp2 = recovery.dropna(subset=["Date", "Sigmoid with Nap"]).sort_values("Date").tail(14)
            last_recovery_nap = tmp2["Sigmoid with Nap"].iloc[-1] if not tmp2.empty else None
            if tmp2.empty:
                st.info("Recovery CSV loaded but no usable rows.")
            else:
                fig, ax = plt.subplots(figsize=(7, 3))
                tmp2_avg = tmp2["Sigmoid with Nap"].mean()
                sld_sig_nap = st.slider(
                    "Select days for moving average",
                    2,
                    11,
                    5,
                    1,
                    width=250,
                    key="sig_nap_ma_slider",
                )
                ax.plot(
                    tmp2["Date"],
                    tmp2["Sigmoid with Nap"],
                    marker="o",
                    markersize=3,
                    color="seagreen",
                )
                roll_avg_recovery_nap = tmp2["Sigmoid with Nap"].rolling(window=sld_sig_nap).mean()
                ax.plot(
                    tmp2["Date"],
                    roll_avg_recovery_nap,
                    color="orange",
                    linestyle="--",
                    alpha=0.25,
                    label=f"MA {sld_sig_nap} days {roll_avg_recovery_nap.iloc[-1]:.2f}",
                )
                ax.axhline(
                    tmp2_avg, color="blue", linestyle=":", alpha=0.6, label=f"Avg {tmp2_avg:.2f}"
                )
                ax.set_xlabel("")
                ax.set_ylim(0, 1)
                ax.legend(loc="lower left")
                ax.tick_params(axis="x", rotation=45, labelsize=6)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                ax.set_ylabel("Sigmoid")
                sns.despine(ax=ax)
                st.pyplot(fig)
        else:
            st.info("No recovery data uploaded yet.")
    with st.expander("Recovery Insights", icon="🧠", expanded=False):
        vol_no_nap = recovery["Sigmoid Recovery Score"].tail(14).std()
        vol_nap = recovery["Sigmoid with Nap"].tail(14).std()
        delta_window = st.slider(
            "Select days window for Δ average", 2, 14, 7, 1, key="delta_avg_window", width=200
        )
        recovery_naps = (
            recovery.dropna(subset=["Date", "DELTA_NAP"]).sort_values("Date").tail(delta_window)
        )
        avg_delta = recovery_naps["DELTA_NAP"].mean() if not recovery_naps.empty else None
        st.write(
            f"Average Δ {delta_window} days: {avg_delta:.2f}"
            if avg_delta is not None
            else "Average Δ not available"
        )
        if last_sigmoid is not None and last_sigmoid >= tmp_avg:
            st.write(f"14-day Last Recovery without Nap: {last_sigmoid:.2f} above Avg")
        else:
            st.write(f"14-day Last Recovery without Nap: {last_sigmoid_nap:.2f} below Avg")
        if last_recovery_nap is not None and last_recovery_nap >= tmp2_avg:
            st.write(f"14-day Last Recovery with Nap: {last_recovery_nap:.2f} above Avg")
        else:
            st.write(f"14-day Last Recovery with Nap: {last_recovery_nap:.2f} below Avg")
        st.write(f"Volatility (STD) without Nap (14 days): {vol_no_nap:.4f}")
        st.write(f"Volatility (STD) with Nap (14 days): {vol_nap:.4f}")

    st.caption(
        "Note: Recovery with Nap may be higher than without nap depending on nap effect.",
        help="Nap effect is computed based on the duration of the nap and the hour it was taken. A positive nap effect indicates that the nap contributed positively to recovery, while a negative effect suggests it may have disrupted sleep patterns.",
    )
    st.markdown("---")
    st.subheader("😴 Sleep score (last 14 days)")
    sleep_score_col = (
        pick_col(recovery, ["Score", "Sleep Score", "SleepScore", "SCORE", "Score.1", "Score.2"])
        if recovery is not None
        else None
    )

    if recovery is not None and sleep_score_col is not None and "Date" in recovery.columns:
        tmp = recovery.dropna(subset=["Date", sleep_score_col]).sort_values("Date").tail(14)
        if tmp.empty:
            st.info("Sleep CSV loaded but no usable rows.")
        else:
            fig, ax = plt.subplots(figsize=(7, 3))
            sld2 = st.slider(
                "Select days for moving average", 2, 11, 5, 1, key="sleep_ma_slider", width=250
            )
            roll_avg_sleep = tmp[sleep_score_col].rolling(window=sld2).mean()
            tmp_avg_sleep = tmp[sleep_score_col].mean()
            ax.axhline(
                tmp_avg_sleep,
                color="blue",
                linestyle=":",
                alpha=0.6,
                label=f"Avg {tmp_avg_sleep:.0f}",
            )
            ax.plot(tmp["Date"], tmp[sleep_score_col], marker="o", markersize=3, color="purple")
            ax.plot(
                tmp["Date"],
                roll_avg_sleep,
                color="orange",
                linestyle="--",
                alpha=0.4,
                label=f"MA {sld2} days {roll_avg_sleep.iloc[-1]:.0f}",
            )
            ax.set_xlabel("")
            ax.legend(loc="lower left")
            ax.set_ylim(50, 100)
            ax.set_ylabel(sleep_score_col)
            ax.tick_params(axis="x", rotation=45, labelsize=6)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            sns.despine(ax=ax)
            st.pyplot(fig)
    else:
        st.info("No sleep data uploaded yet (or score column not detected).")
    st.markdown("---")

    with st.expander("🛠 Debug"):
        st.write("Workouts cols:", list(workouts.columns))
        st.write("Sleep cols:", list(sleep.columns))
        st.write("Recovery cols:", list(recovery.columns))
    st.markdown("--")

    return sleep
