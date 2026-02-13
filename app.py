import pandas as pd
import streamlit as st
from pathlib import Path
import base64


st.set_page_config(page_title="Análisis de Pérdidas y Ganancias", layout="wide")


def get_basic_auth_credentials() -> tuple[str, str]:
    username = "admin"
    password = "admin123"

    if "basic_auth" in st.secrets:
        username = st.secrets["basic_auth"].get("username", username)
        password = st.secrets["basic_auth"].get("password", password)

    username = st.secrets.get("BASIC_AUTH_USERNAME", username)
    password = st.secrets.get("BASIC_AUTH_PASSWORD", password)
    return str(username), str(password)


def init_auth_state() -> None:
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = ""


def render_login_gate() -> None:
    valid_user, valid_password = get_basic_auth_credentials()
    init_auth_state()

    if st.session_state["authenticated"]:
        with st.sidebar:
            st.caption(f"Sesión: {st.session_state['auth_user']}")
            if st.button("Cerrar sesión"):
                st.session_state["authenticated"] = False
                st.session_state["auth_user"] = ""
                st.rerun()
        return

    st.title("Acceso al Dashboard")
    st.caption("Ingresa tus credenciales para continuar.")
    with st.form("login_form", clear_on_submit=False):
        user_input = st.text_input("Usuario")
        password_input = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Ingresar")

    if submitted:
        if user_input == valid_user and password_input == valid_password:
            st.session_state["authenticated"] = True
            st.session_state["auth_user"] = user_input
            st.rerun()
        else:
            st.error("Credenciales inválidas.")

    st.stop()


@st.cache_data
def load_financial_data(path: str) -> pd.DataFrame:
    df = pd.read_pickle(path).copy()
    df["AÑO"] = pd.to_numeric(df["AÑO"], errors="coerce").astype("Int64")
    df["NOMBRE"] = df["NOMBRE"].astype(str).str.strip()
    return df


def style_income_statement_row(row: pd.Series, total_rows: set[int], detail_rows: set[int]) -> list[str]:
    if row.name in total_rows:
        return ["border-top: 2px solid #1f2937; font-weight: 700; background-color: #f5f7fb;"] * len(row)
    if row.name in detail_rows:
        return ["color: #374151;"] * len(row)
    return [""] * len(row)


render_login_gate()
header_col, logo_col = st.columns([5.5, 1.5], vertical_alignment="top")
with header_col:
    st.title("Análisis de Pérdidas y Ganancias")
    st.caption("Datos de Supercias 2021-2024")
with logo_col:
    logo_path = Path("Logo Andersen.png")
    if logo_path.exists():
        logo_base64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
        st.markdown(
            f"""
            <div style="margin-top: -42px;">
                <img src="data:image/png;base64,{logo_base64}" style="width: 100%; height: auto; display: block;" />
            </div>
            """,
            unsafe_allow_html=True,
        )

data = load_financial_data("supercias_resultados.pkl")

statement_structure = [
    {"column": "INGRESOS", "label": "INGRESOS", "sign": "", "is_total": False, "is_detail": False},
    {"column": "COSTO_VENTAS", "label": "COSTO_VENTAS", "sign": "(-)", "is_total": False, "is_detail": False},
    {"column": "COSTO_DIST_LOG", "label": "COSTO_DIST_LOG", "sign": "(-)", "is_total": False, "is_detail": False},
    {"column": "CONTRIBUCION_MARGINAL", "label": "CONTRIBUCION_MARGINAL", "sign": "=", "is_total": True, "is_detail": False},
    {"column": "GASTO_OPERACIONAL", "label": "GASTO_OPERACIONAL", "sign": "(-)", "is_total": False, "is_detail": False},
    {"column": "GASTO_DE_PERSONAL", "label": "GASTO_DE_PERSONAL", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO_ARRENDAMIENTO", "label": "GASTO_ARRENDAMIENTO", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO_PUBLICIDAD", "label": "GASTO_PUBLICIDAD", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO_SERVICIOS_BASICOS", "label": "GASTO_SERVICIOS_BASICOS", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO_GESTION", "label": "GASTO_GESTION", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO_MANTENIMIENTO", "label": "GASTO_MANTENIMIENTO", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO_SERVICIOS_PROFESIONALES", "label": "GASTO_SERVICIOS_PROFESIONALES", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO_SEGUROS", "label": "GASTO_SEGUROS", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO_OTROS", "label": "GASTO_OTROS", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "UTILIDAD_OPERACIONAL", "label": "UTILIDAD_OPERACIONAL", "sign": "=", "is_total": True, "is_detail": False},
    {"column": "GASTOS_ADMINISTRATIVOS", "label": "GASTOS_ADMINISTRATIVOS", "sign": "(-)", "is_total": False, "is_detail": False},
    {"column": "EBITDA", "label": "EBITDA", "sign": "=", "is_total": True, "is_detail": False},
    {"column": "DEPRECIACION", "label": "DEPRECIACION", "sign": "(+)", "is_total": False, "is_detail": False},
    {"column": "AMORTIZACION", "label": "AMORTIZACION", "sign": "(+)", "is_total": False, "is_detail": False},
    {"column": "COSTO_FINANCIERO", "label": "COSTO_FINANCIERO", "sign": "(-)", "is_total": False, "is_detail": False},
    {"column": "RESULTADO_ANTES_IMPUESTOS", "label": "RESULTADO_ANTES_IMPUESTOS", "sign": "=", "is_total": True, "is_detail": False},
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
    pct_income_rows = {
        "INGRESOS",
        "CONTRIBUCION_MARGINAL",
        "GASTO_OPERACIONAL",
        "UTILIDAD_OPERACIONAL",
        "GASTOS_ADMINISTRATIVOS",
        "EBITDA",
        "RESULTADO_ANTES_IMPUESTOS",
    }

    # Si hay múltiples registros por año/empresa, consolidamos por suma.
    annual_df = (
        company_df.groupby("AÑO", dropna=False)[available_columns]
        .sum(numeric_only=True)
        .reindex([2021, 2022, 2023, 2024])
    )

    ruc = company_df["RUC"].dropna().astype(str).iloc[0] if not company_df["RUC"].dropna().empty else "-"
    st.subheader(selected_company)
    st.write(f"**RUC:** {ruc}")
    ingresos_2024 = annual_df.loc[2024, "INGRESOS"] if "INGRESOS" in annual_df.columns else pd.NA

    report_rows = []
    total_rows = set()
    detail_rows = set()
    for idx, item in enumerate(statement_structure):
        column = item["column"]
        value_2021 = annual_df.loc[2021, column] if column in annual_df.columns else pd.NA
        value_2022 = annual_df.loc[2022, column] if column in annual_df.columns else pd.NA
        value_2023 = annual_df.loc[2023, column] if column in annual_df.columns else pd.NA
        value_2024 = annual_df.loc[2024, column] if column in annual_df.columns else pd.NA

        var_abs = pd.NA
        var_pct = pd.NA
        pct_ingresos = pd.NA
        if pd.notna(value_2023) and pd.notna(value_2024):
            var_abs = value_2024 - value_2023
            var_pct = (var_abs / value_2023 * 100) if value_2023 != 0 else pd.NA
        if (
            column in pct_income_rows
            and pd.notna(value_2024)
            and pd.notna(ingresos_2024)
            and ingresos_2024 != 0
        ):
            pct_ingresos = (value_2024 / ingresos_2024) * 100

        if item["is_total"]:
            total_rows.add(idx)
        if item["is_detail"]:
            detail_rows.add(idx)

        display_label = f"{item['sign']} {item['label']}".strip()
        report_rows.append(
            {
                "Cuenta": display_label,
                "2021": value_2021,
                "2022": value_2022,
                "2023": value_2023,
                "2024": value_2024,
                "% Ingresos": pct_ingresos,
                "Variación 2025/2024": var_abs,
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
                "2021": "{:,.0f}",
                "2022": "{:,.0f}",
                "2023": "{:,.0f}",
                "2024": "{:,.0f}",
                "% Ingresos": "{:.2f}%",
                "Variación 2025/2024": "{:,.0f}",
                "Variación %": "{:.2f}%",
            },
            na_rep="-",
        )
        .apply(lambda row: style_income_statement_row(row, total_rows, detail_rows), axis=1)
    )

    st.dataframe(styled_report, use_container_width=True, hide_index=True)
else:
    st.info("Selecciona una empresa para visualizar su estado de resultados.")
