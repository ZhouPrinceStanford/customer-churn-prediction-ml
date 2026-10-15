
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Churn Business Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn | Business Intelligence")
st.caption("Turning customer data into retention decisions")

@st.cache_data
def load_data():
    df = pd.read_csv("Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df

df = load_data()

churned = df[df["Churn"].astype(str).str.strip().str.lower() == "yes"]
active = df[df["Churn"].astype(str).str.strip().str.lower() == "no"]

total_customers = len(df)
churn_rate = len(churned) / total_customers if total_customers else 0
revenue_at_risk = churned["MonthlyCharges"].sum()
average_churn_tenure = churned["tenure"].mean()

a, b, c, d = st.columns(4)

a.metric("Total customers", f"{total_customers:,}")
b.metric("Customers lost", f"{len(churned):,}")
c.metric("Churn rate", f"{churn_rate:.1%}")
d.metric("Monthly charges lost", f"${revenue_at_risk:,.0f}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Churn by contract")
    contract_data = (
        df.groupby(["Contract", "Churn"])
        .size()
        .unstack(fill_value=0)
    )
    if "Yes" in contract_data.columns:
        st.bar_chart(contract_data["Yes"])
    else:
        st.info("No churn records found.")

with right:
    st.subheader("Churn by tenure")
    tenure_data = df.copy()
    tenure_data["Tenure group"] = pd.cut(
        tenure_data["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0–12 months", "13–24 months",
                "25–48 months", "49–72 months"]
    )
    grouped = (
        tenure_data.groupby(["Tenure group", "Churn"],
                            observed=False)
        .size()
        .unstack(fill_value=0)
    )
    if "Yes" in grouped.columns:
        st.bar_chart(grouped["Yes"])
    else:
        st.info("No churn records found.")

st.divider()

st.subheader("Churn by payment method")
payment_data = (
    df.groupby(["PaymentMethod", "Churn"])
    .size()
    .unstack(fill_value=0)
)
if "Yes" in payment_data.columns:
    st.bar_chart(payment_data["Yes"])

st.subheader("Business insights")

if churn_rate > 0:
    st.write(
        f"- **Retention:** {churn_rate:.1%} of customers have churned."
    )
    st.write(
        f"- **Revenue exposure:** churned customers represented "
        f"${revenue_at_risk:,.2f} in monthly charges."
    )
    st.write(
        f"- **Customer tenure:** churned customers stayed for an "
        f"average of {average_churn_tenure:.1f} months."
    )

st.markdown("""
**Recommended actions**
1. Investigate why month-to-month customers leave.
2. Test targeted retention offers for high-risk customers.
3. Compare payment methods and customer tenure before choosing
   retention campaigns.
4. Measure whether each campaign reduces churn and protects revenue.

*Revenue at risk is an estimate based on churned customers' monthly
charges, not a forecast of recoverable revenue.*
""")

st.subheader("Explore customer records")
contract_filter = st.selectbox(
    "Filter by contract",
    ["All"] + sorted(df["Contract"].dropna().unique().tolist())
)

filtered = df if contract_filter == "All" else df[
    df["Contract"] == contract_filter
]

st.dataframe(filtered, use_container_width=True)
st.download_button(
    "Download filtered data",
    filtered.to_csv(index=False),
    file_name="churn_customer_data.csv",
    mime="text/csv"
)
