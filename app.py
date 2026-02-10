import pandas as pd
import streamlit as st


st.set_page_config(page_title="Dashboard - Estado de Resultados", layout="wide")


@st.cache_data
def load_financial_data(path: str) -> pd.DataFrame:
    df = pd.read_pickle(path).copy()
    df["AÑO"] = pd.to_numeric(df["AÑO"], errors="coerce").astype("Int64")
    df["NOMBRE"] = df["NOMBRE"].astype(str).str.strip()
    return df


def style_income_statement_row(row: pd.Series, total_rows: set[int]) -> list[str]:
    if row.name in total_rows:
        return ["border-top: 2px solid #1f2937; font-weight: 700; background-color: #f5f7fb;"] * len(row)
    return [""] * len(row)


st.title("Estado de Resultados por Empresa")
st.caption("Datos de Supercias 2023-2024")

data = load_financial_data("supercias.pkl")

statement_structure = [
    {"column": "INGRESOS", "label": "INGRESOS", "sign": "", "is_total": False},
    {"column": "COSTO_VENTAS", "label": "COSTO_VENTAS", "sign": "(-)", "is_total": False},
    {"column": "COSTO_DIST_LOG", "label": "COSTO_DIST_LOG", "sign": "(-)", "is_total": False},
    {"column": "CONTRIBUCION_MARGINAL", "label": "CONTRIBUCION_MARGINAL", "sign": "=", "is_total": True},
    {"column": "GASTO_OPERACIONAL", "label": "GASTO_OPERACIONAL", "sign": "(-)", "is_total": False},
    {"column": "UTILIDAD_OPERACIONAL", "label": "UTILIDAD_OPERACIONAL", "sign": "=", "is_total": True},
    {"column": "GASTOS_ADMINISTRATIVOS", "label": "GASTOS_ADMINISTRATIVOS", "sign": "(-)", "is_total": False},
    {"column": "EBITDA", "label": "EBITDA", "sign": "=", "is_total": True},
    {"column": "DEPRECIACION", "label": "DEPRECIACION", "sign": "(+)", "is_total": False},
    {"column": "AMORTIZACION", "label": "AMORTIZACION", "sign": "(+)", "is_total": False},
    {"column": "COSTO_FINANCIERO", "label": "COSTO_FINANCIERO", "sign": "(-)", "is_total": False},
    {"column": "RESULTADO_ANTES_IMPUESTOS", "label": "RESULTADO_ANTES_IMPUESTOS", "sign": "=", "is_total": True},
]

company_options = sorted(data["NOMBRE"].dropna().unique().tolist())
selected_company = st.selectbox(
    "Buscar y seleccionar empresa",
    options=company_options,
    index=None,
    placeholder="Escribe el nombre de la empresa...",
)

if selected_company:
    company_df = data[data["NOMBRE"] == selected_company].copy()
    statement_columns = [item["column"] for item in statement_structure]
    available_columns = [col for col in statement_columns if col in company_df.columns]
    missing_columns = [col for col in statement_columns if col not in company_df.columns]

    # Si hay múltiples registros por año/empresa, consolidamos por suma.
    annual_df = (
        company_df.groupby("AÑO", dropna=False)[available_columns]
        .sum(numeric_only=True)
        .reindex([2023, 2024])
    )

    ruc = company_df["RUC"].dropna().astype(str).iloc[0] if not company_df["RUC"].dropna().empty else "-"
    st.subheader(selected_company)
    st.write(f"**RUC:** {ruc}")

    report_rows = []
    total_rows = set()
    for idx, item in enumerate(statement_structure):
        column = item["column"]
        value_2023 = annual_df.loc[2023, column] if column in annual_df.columns else pd.NA
        value_2024 = annual_df.loc[2024, column] if column in annual_df.columns else pd.NA

        var_abs = pd.NA
        var_pct = pd.NA
        if pd.notna(value_2023) and pd.notna(value_2024):
            var_abs = value_2024 - value_2023
            var_pct = (var_abs / value_2023 * 100) if value_2023 != 0 else pd.NA

        if item["is_total"]:
            total_rows.add(idx)

        display_label = f"{item['sign']} {item['label']}".strip()
        report_rows.append(
            {
                "Cuenta": display_label,
                "2023": value_2023,
                "2024": value_2024,
                "Variación": var_abs,
                "Variación %": var_pct,
            }
        )

    report_df = pd.DataFrame(report_rows)
    if missing_columns:
        st.warning(f"Columnas no encontradas en el dataset: {', '.join(missing_columns)}")

    styled_report = (
        report_df.style
        .format(
            {
                "2023": "{:,.0f}",
                "2024": "{:,.0f}",
                "Variación": "{:,.0f}",
                "Variación %": "{:.2f}%",
            },
            na_rep="-",
        )
        .apply(lambda row: style_income_statement_row(row, total_rows), axis=1)
    )

    st.dataframe(styled_report, use_container_width=True, hide_index=True)
else:
    st.info("Selecciona una empresa para visualizar su estado de resultados.")
