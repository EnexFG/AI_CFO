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


@st.cache_data
def load_balance_data(path_2023: str, path_2024: str) -> pd.DataFrame:
    df_2023 = pd.read_pickle(path_2023).copy()
    df_2024 = pd.read_pickle(path_2024).copy()

    df_2023["AÑO"] = 2023
    df_2024["AÑO"] = 2024

    for df in [df_2023, df_2024]:
        df["NOMBRE"] = df["NOMBRE"].astype(str).str.strip()
        df["RUC"] = df["RUC"].astype(str).str.strip()

    return pd.concat([df_2023, df_2024], ignore_index=True)


def style_income_statement_row(row: pd.Series, total_rows: set[int], detail_rows: set[int]) -> list[str]:
    if row.name in total_rows:
        return ["border-top: 2px solid #1f2937; font-weight: 700; background-color: #f5f7fb;"] * len(row)
    if row.name in detail_rows:
        return ["color: #374151;"] * len(row)
    return [""] * len(row)


def style_hierarchy_label(row: pd.Series, level_map: dict[int, int]) -> list[str]:
    level = level_map.get(row.name, 0)
    if level == 2:
        return ["padding-left: 2rem; color: #4b5563;"]
    if level == 1:
        return ["padding-left: 1rem;"]
    return [""]


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
balance_data = load_balance_data("supercias_balances_2023.pkl", "supercias_balances_2024.pkl")

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
    st.markdown("#### Estado de Resultados")

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

    st.markdown("#### Análisis de Balance General")
    balance_structure = [
        {"column": "ACTIVO", "label": "ACTIVO", "sign": "=", "level": 0, "is_total": True, "is_detail": False},
        {"column": "ACTIVO CORRIENTE", "label": "ACTIVO CORRIENTE", "sign": "->", "level": 1, "is_total": False, "is_detail": True},
        {"column": "DISPONIBLE", "label": "DISPONIBLE", "sign": "-->", "level": 2, "is_total": False, "is_detail": True},
        {"column": "INVERSIONES", "label": "INVERSIONES", "sign": "-->", "level": 2, "is_total": False, "is_detail": True},
        {"column": "CUENTAS POR COBRAR CLIENTES", "label": "CUENTAS POR COBRAR CLIENTES", "sign": "-->", "level": 2, "is_total": False, "is_detail": True},
        {"column": "CUENTAS POR COBRAR RELACIONADAS", "label": "CUENTAS POR COBRAR RELACIONADAS", "sign": "-->", "level": 2, "is_total": False, "is_detail": True},
        {"column": "PROVISIÓN INCOBRABLE Y DETERIORO", "label": "PROVISIÓN INCOBRABLE Y DETERIORO", "sign": "-->", "level": 2, "is_total": False, "is_detail": True},
        {"column": "INVENTARIO", "label": "INVENTARIO", "sign": "-->", "level": 2, "is_total": False, "is_detail": True},
        {"column": "CRÉDITO TRIBUTARIO", "label": "CRÉDITO TRIBUTARIO", "sign": "-->", "level": 2, "is_total": False, "is_detail": True},
        {"column": "OTROS ACTIVOS CORRIENTES", "label": "OTROS ACTIVOS CORRIENTES", "sign": "-->", "level": 2, "is_total": False, "is_detail": True},
        {"column": "ACTIVO NO CORRIENTE", "label": "ACTIVO NO CORRIENTE", "sign": "->", "level": 1, "is_total": False, "is_detail": True},
        {"column": "PASIVO", "label": "PASIVO", "sign": "=", "level": 0, "is_total": True, "is_detail": False},
        {"column": "PASIVO CORRIENTE", "label": "PASIVO CORRIENTE", "sign": "->", "level": 1, "is_total": False, "is_detail": True},
        {"column": "PASIVO NO CORRIENTE", "label": "PASIVO NO CORRIENTE", "sign": "->", "level": 1, "is_total": False, "is_detail": True},
        {"column": "PATRIMONIO", "label": "PATRIMONIO", "sign": "=", "level": 0, "is_total": True, "is_detail": False},
    ]
    balance_columns = [item["column"] for item in balance_structure]
    balance_company_df = balance_data[balance_data["RUC"] == str(ruc)].copy()
    if balance_company_df.empty:
        balance_company_df = balance_data[balance_data["NOMBRE"] == selected_company].copy()

    if balance_company_df.empty:
        st.warning("No se encontró información de balance general para esta empresa.")
    else:
        available_balance_cols = [col for col in balance_columns if col in balance_company_df.columns]
        missing_balance_cols = [col for col in balance_columns if col not in balance_company_df.columns]
        annual_balance_df = (
            balance_company_df.groupby("AÑO", dropna=False)[available_balance_cols]
            .sum(numeric_only=True)
            .reindex([2023, 2024])
        )

        balance_rows = []
        balance_total_rows = set()
        balance_detail_rows = set()
        balance_level_map = {}
        vertical_parent = {
            "ACTIVO": "ACTIVO",
            "ACTIVO CORRIENTE": "ACTIVO",
            "DISPONIBLE": "ACTIVO CORRIENTE",
            "INVERSIONES": "ACTIVO CORRIENTE",
            "CUENTAS POR COBRAR CLIENTES": "ACTIVO CORRIENTE",
            "CUENTAS POR COBRAR RELACIONADAS": "ACTIVO CORRIENTE",
            "PROVISIÓN INCOBRABLE Y DETERIORO": "ACTIVO CORRIENTE",
            "INVENTARIO": "ACTIVO CORRIENTE",
            "CRÉDITO TRIBUTARIO": "ACTIVO CORRIENTE",
            "OTROS ACTIVOS CORRIENTES": "ACTIVO CORRIENTE",
            "ACTIVO NO CORRIENTE": "ACTIVO",
            "PASIVO": "PASIVO",
            "PASIVO CORRIENTE": "PASIVO",
            "PASIVO NO CORRIENTE": "PASIVO",
            "PATRIMONIO": "PATRIMONIO",
        }
        for idx, item in enumerate(balance_structure):
            col = item["column"]
            value_2023 = annual_balance_df.loc[2023, col] if col in annual_balance_df.columns else pd.NA
            value_2024 = annual_balance_df.loc[2024, col] if col in annual_balance_df.columns else pd.NA

            var_abs = pd.NA
            var_pct = pd.NA
            vertical_pct = pd.NA
            if pd.notna(value_2023) and pd.notna(value_2024):
                var_abs = value_2024 - value_2023
                var_pct = (var_abs / value_2023 * 100) if value_2023 != 0 else pd.NA
            parent_col = vertical_parent.get(col)
            if parent_col and parent_col in annual_balance_df.columns and pd.notna(value_2024):
                parent_value_2024 = annual_balance_df.loc[2024, parent_col]
                if pd.notna(parent_value_2024) and parent_value_2024 != 0:
                    vertical_pct = (value_2024 / parent_value_2024) * 100

            if item["is_total"]:
                balance_total_rows.add(idx)
            if item["is_detail"]:
                balance_detail_rows.add(idx)
            balance_level_map[idx] = item.get("level", 0)

            display_label = f"{item['sign']} {item['label']}".strip()

            balance_rows.append(
                {
                    "Cuenta": display_label,
                    "2023": value_2023,
                    "2024": value_2024,
                    "Variación 2024/2023": var_abs,
                    "Comparativo Horizontal": var_pct,
                    "Comparativo Vertical": vertical_pct,
                }
            )

        balance_df = pd.DataFrame(balance_rows)
        if missing_balance_cols:
            st.warning(f"Columnas de balance no encontradas: {', '.join(missing_balance_cols)}")

        styled_balance = (
            balance_df.style
            .format(
                {
                    "2023": "{:,.0f}",
                    "2024": "{:,.0f}",
                    "Variación 2024/2023": "{:,.0f}",
                    "Comparativo Horizontal": "{:.2f}%",
                    "Comparativo Vertical": "{:.2f}%",
                },
                na_rep="-",
            )
            .apply(lambda row: style_income_statement_row(row, balance_total_rows, balance_detail_rows), axis=1)
            .apply(lambda row: style_hierarchy_label(row, balance_level_map), axis=1, subset=["Cuenta"])
        )

        st.dataframe(styled_balance, use_container_width=True, hide_index=True)
else:
    st.info("Selecciona una empresa para visualizar su estado de resultados.")
