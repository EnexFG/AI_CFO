import pandas as pd
import streamlit as st


st.set_page_config(page_title="Dashboard - Estado de Resultados", layout="wide")


@st.cache_data
def load_financial_data(path: str) -> pd.DataFrame:
    df = pd.read_pickle(path).copy()
    df["AÑO"] = pd.to_numeric(df["AÑO"], errors="coerce").astype("Int64")
    df["NOMBRE"] = df["NOMBRE"].astype(str).str.strip()
    return df


def style_income_statement_row(row: pd.Series) -> list[str]:
    if row["Código"] == "vc61":
        return [
            "border-top: 2px solid #1f2937; font-weight: 700; background-color: #f5f7fb;"
        ] * len(row)
    return [""] * len(row)


st.title("Estado de Resultados por Empresa")
st.caption("Datos de Supercias 2023-2024")

data = load_financial_data("supercias.pkl")

# Variables definidas internamente (sin depender de Excel en runtime).
variables = ["vx50", "v69", "vc61"]
labels = {
    "vx50": "Ingresos",
    "v69": "(-) Gastos",
    "vc61": "= Utilidad del ejercicio",
}

company_options = sorted(data["NOMBRE"].dropna().unique().tolist())
selected_company = st.selectbox(
    "Buscar y seleccionar empresa",
    options=company_options,
    index=None,
    placeholder="Escribe el nombre de la empresa...",
)

if selected_company:
    company_df = data[data["NOMBRE"] == selected_company].copy()

    # Si hay múltiples registros por año/empresa, consolidamos por suma.
    annual_df = (
        company_df.groupby("AÑO", dropna=False)[variables]
        .sum(numeric_only=True)
        .reindex([2023, 2024])
    )

    ruc = company_df["RUC"].dropna().astype(str).iloc[0] if not company_df["RUC"].dropna().empty else "-"
    st.subheader(selected_company)
    st.write(f"**RUC:** {ruc}")

    report_rows = []
    for var in variables:
        value_2023 = annual_df.loc[2023, var] if 2023 in annual_df.index else pd.NA
        value_2024 = annual_df.loc[2024, var] if 2024 in annual_df.index else pd.NA

        var_abs = pd.NA
        var_pct = pd.NA
        if pd.notna(value_2023) and pd.notna(value_2024):
            var_abs = value_2024 - value_2023
            var_pct = (var_abs / value_2023 * 100) if value_2023 != 0 else pd.NA

        report_rows.append(
            {
                "Cuenta": labels.get(var, var.upper()),
                "Código": var,
                "2023": value_2023,
                "2024": value_2024,
                "Variación": var_abs,
                "Variación %": var_pct,
            }
        )

    report_df = pd.DataFrame(report_rows)
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
        .apply(style_income_statement_row, axis=1)
    )

    st.table(styled_report.hide(axis="index"))
else:
    st.info("Selecciona una empresa para visualizar su estado de resultados.")
