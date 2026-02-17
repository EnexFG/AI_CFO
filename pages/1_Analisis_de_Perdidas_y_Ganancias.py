import streamlit as st

from dashboard_core import (
    get_company_ruc,
    load_financial_data,
    render_company_identity,
    render_company_selector,
    render_header,
    render_income_statement,
    render_login_gate,
)


st.set_page_config(page_title="Analisis de Perdidas y Ganancias", layout="wide")
render_login_gate()
render_header()

results_data = load_financial_data("supercias_resultados.pkl")
selected_company = render_company_selector(results_data)

st.markdown("#### Analisis de Perdidas y Ganancias")
if not selected_company:
    st.info("Selecciona una empresa en la barra lateral.")
    st.stop()

company_ruc = get_company_ruc(results_data, selected_company)
render_company_identity(selected_company, company_ruc)
render_income_statement(results_data, selected_company)
