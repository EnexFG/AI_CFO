import base64
import io
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Andersen Intelligence", layout="wide")


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
            st.caption(f"Sesion: {st.session_state['auth_user']}")
            if st.button("Cerrar sesion"):
                st.session_state["authenticated"] = False
                st.session_state["auth_user"] = ""
                st.rerun()
        return

    st.title("Acceso al Dashboard")
    st.caption("Ingresa tus credenciales para continuar.")
    with st.form("login_form", clear_on_submit=False):
        user_input = st.text_input("Usuario")
        password_input = st.text_input("Contrasena", type="password")
        submitted = st.form_submit_button("Ingresar")

    if submitted:
        if user_input == valid_user and password_input == valid_password:
            st.session_state["authenticated"] = True
            st.session_state["auth_user"] = user_input
            st.rerun()
        else:
            st.error("Credenciales invalidas.")

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
        return ["padding-left: 2.4rem; color: #4b5563; font-weight: 500;"]
    if level == 1:
        return ["padding-left: 1.2rem; font-weight: 600;"]
    return [""]


def style_balance_row(row: pd.Series, total_rows: set[int], level_map: dict[int, int]) -> list[str]:
    if row.name in total_rows:
        return ["border-top: 2px solid #1f2937; font-weight: 700; background-color: #eef2ff;"] * len(row)
    level = level_map.get(row.name, 0)
    if level == 1:
        return ["background-color: #f8fafc;"] * len(row)
    if level == 2:
        return ["background-color: #fcfcfd; color: #4b5563;"] * len(row)
    return [""] * len(row)


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    for engine in ("xlsxwriter", "openpyxl"):
        try:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine=engine) as writer:
                df.to_excel(writer, index=False, sheet_name=sheet_name)
            buffer.seek(0)
            return buffer.getvalue()
        except Exception:
            continue

    raise RuntimeError("No hay motor de Excel disponible (xlsxwriter/openpyxl).")


def safe_filename(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(text)).strip("_")


render_login_gate()
header_col, logo_col = st.columns([5.5, 1.5], vertical_alignment="top")
with header_col:
    st.title("Andersen Intelligence")
    st.markdown(
        "Inteligencia de datos financieros que convierte informacion contable en insights accionables para evaluar desempeno, estructura y riesgo empresarial."
    )
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

    annual_df = (
        company_df.groupby("AÑO", dropna=False)[available_columns]
        .sum(numeric_only=True)
        .reindex([2021, 2022, 2023, 2024])
    )

    ruc = company_df["RUC"].dropna().astype(str).iloc[0] if not company_df["RUC"].dropna().empty else "-"
    st.subheader(selected_company)
    st.write(f"**RUC:** {ruc}")
    ingresos_2024 = annual_df.loc[2024, "INGRESOS"] if "INGRESOS" in annual_df.columns else pd.NA
    st.markdown("#### Analisis de Perdidas y Ganancias")

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
                "Variacion 2025/2024": var_abs,
                "Variacion %": var_pct,
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
                "Variacion 2025/2024": "{:,.0f}",
                "Variacion %": "{:.2f}%",
            },
            na_rep="-",
        )
        .apply(lambda row: style_income_statement_row(row, total_rows, detail_rows), axis=1)
    )

    st.dataframe(styled_report, width="stretch", hide_index=True)
    report_export_df = report_df.copy()
    try:
        report_bytes = to_excel_bytes(report_export_df, "Perdidas_Ganancias")
        report_file = f"analisis_perdidas_ganancias_{safe_filename(selected_company)}.xlsx"
        st.download_button(
            "Descargar Excel - Perdidas y Ganancias",
            data=report_bytes,
            file_name=report_file,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"download_pyg_{safe_filename(selected_company)}",
        )
    except RuntimeError:
        st.error("No se pudo generar Excel para Perdidas y Ganancias (falta xlsxwriter/openpyxl).")

    st.markdown("#### Analisis de Balance General")
    balance_structure = [
        {"column": "ACTIVO", "label": "ACTIVO", "level": 0, "is_total": True, "is_detail": False},
        {"column": "ACTIVO CORRIENTE", "label": "ACTIVO CORRIENTE", "level": 1, "is_total": False, "is_detail": True},
        {"column": "DISPONIBLE", "label": "DISPONIBLE", "level": 2, "is_total": False, "is_detail": True},
        {"column": "INVERSIONES", "label": "INVERSIONES", "level": 2, "is_total": False, "is_detail": True},
        {"column": "CUENTAS POR COBRAR CLIENTES", "label": "CUENTAS POR COBRAR CLIENTES", "level": 2, "is_total": False, "is_detail": True},
        {"column": "CUENTAS POR COBRAR RELACIONADAS", "label": "CUENTAS POR COBRAR RELACIONADAS", "level": 2, "is_total": False, "is_detail": True},
        {"column": "PROVISIÓN INCOBRABLE Y DETERIORO", "label": "PROVISIÓN INCOBRABLE Y DETERIORO", "level": 2, "is_total": False, "is_detail": True},
        {"column": "INVENTARIO", "label": "INVENTARIO", "level": 2, "is_total": False, "is_detail": True},
        {"column": "CRÉDITO TRIBUTARIO", "label": "CRÉDITO TRIBUTARIO", "level": 2, "is_total": False, "is_detail": True},
        {"column": "OTROS ACTIVOS CORRIENTES", "label": "OTROS ACTIVOS CORRIENTES", "level": 2, "is_total": False, "is_detail": True},
        {"column": "ACTIVO NO CORRIENTE", "label": "ACTIVO NO CORRIENTE", "level": 1, "is_total": False, "is_detail": True},
        {"column": "PROPIEDAD, PLANTA Y EQUIPO", "label": "PROPIEDAD, PLANTA Y EQUIPO", "level": 2, "is_total": False, "is_detail": True},
        {"column": "PROPIEDADES DE INVERSIÓN", "label": "PROPIEDADES DE INVERSIÓN", "level": 2, "is_total": False, "is_detail": True},
        {"column": "ACTIVOS BIOLÓGICOS", "label": "ACTIVOS BIOLÓGICOS", "level": 2, "is_total": False, "is_detail": True},
        {"column": "ACTIVO INTANGIBLE", "label": "ACTIVO INTANGIBLE", "level": 2, "is_total": False, "is_detail": True},
        {"column": "ACTIVOS POR IMPUESTOS DIFERIDOS", "label": "ACTIVOS POR IMPUESTOS DIFERIDOS", "level": 2, "is_total": False, "is_detail": True},
        {"column": "ACTIVOS FINANCIEROS NO CORRIENTES", "label": "ACTIVOS FINANCIEROS NO CORRIENTES", "level": 2, "is_total": False, "is_detail": True},
        {"column": "DERECHO DE USO POR ACTIVOS ARRENDADOS", "label": "DERECHO DE USO POR ACTIVOS ARRENDADOS", "level": 2, "is_total": False, "is_detail": True},
        {"column": "DOCUMENTOS Y CUENTAS POR COBRAR NO RELACIONADOS", "label": "DOCUMENTOS Y CUENTAS POR COBRAR NO RELACIONADOS", "level": 2, "is_total": False, "is_detail": True},
        {"column": "DOCUMENTOS Y CUENTAS POR COBRAR RELACIONADOS", "label": "DOCUMENTOS Y CUENTAS POR COBRAR RELACIONADOS", "level": 2, "is_total": False, "is_detail": True},
        {"column": "OTROS ACTIVOS NO CORRIENTES", "label": "OTROS ACTIVOS NO CORRIENTES", "level": 2, "is_total": False, "is_detail": True},
        {"column": "PASIVO", "label": "PASIVO", "level": 0, "is_total": True, "is_detail": False},
        {"column": "PASIVO CORRIENTE", "label": "PASIVO CORRIENTE", "level": 1, "is_total": False, "is_detail": True},
        {"column": "CUENTAS POR PAGAR", "label": "CUENTAS POR PAGAR", "level": 2, "is_total": False, "is_detail": True},
        {"column": "OBLIGACIONES FINACIERAS CORTO PLAZO", "label": "OBLIGACIONES FINACIERAS CORTO PLAZO", "level": 2, "is_total": False, "is_detail": True},
        {"column": "IMPUESTOS POR PAGAR", "label": "IMPUESTOS POR PAGAR", "level": 2, "is_total": False, "is_detail": True},
        {"column": "OTRAS CUENTAS POR PAGAR", "label": "OTRAS CUENTAS POR PAGAR", "level": 2, "is_total": False, "is_detail": True},
        {"column": "PROVISIONES", "label": "PROVISIONES", "level": 2, "is_total": False, "is_detail": True},
        {"column": "OTROS PASIVOS CORRIENTES", "label": "OTROS PASIVOS CORRIENTES", "level": 2, "is_total": False, "is_detail": True},
        {"column": "PASIVO NO CORRIENTE", "label": "PASIVO NO CORRIENTE", "level": 1, "is_total": False, "is_detail": True},
        {"column": "OBLIGACIONES FINANCIERAS LARGO PLAZO", "label": "OBLIGACIONES FINANCIERAS LARGO PLAZO", "level": 2, "is_total": False, "is_detail": True},
        {"column": "ARRENDAMIENTO LARGO PLAZO", "label": "ARRENDAMIENTO LARGO PLAZO", "level": 2, "is_total": False, "is_detail": True},
        {"column": "PASIVO DIFERIDO", "label": "PASIVO DIFERIDO", "level": 2, "is_total": False, "is_detail": True},
        {"column": "BENEFICIOS EMPLEADOS LARGO PLAZO", "label": "BENEFICIOS EMPLEADOS LARGO PLAZO", "level": 2, "is_total": False, "is_detail": True},
        {"column": "OTRAS CUENTAS POR PAGAR LARGO PLAZO", "label": "OTRAS CUENTAS POR PAGAR LARGO PLAZO", "level": 2, "is_total": False, "is_detail": True},
        {"column": "OTROS PASIVOS NO CORRIENTES", "label": "OTROS PASIVOS NO CORRIENTES", "level": 2, "is_total": False, "is_detail": True},
        {"column": "PATRIMONIO", "label": "PATRIMONIO", "level": 0, "is_total": True, "is_detail": False},
        {"column": "CAPITAL", "label": "CAPITAL", "level": 2, "is_total": False, "is_detail": True},
        {"column": "APORTES PARA FUTURA CAPITALIZACIÓN", "label": "APORTES PARA FUTURA CAPITALIZACIÓN", "level": 2, "is_total": False, "is_detail": True},
        {"column": "PRIMA POR EMISIÓN PRIMARIA DE ACCIONES", "label": "PRIMA POR EMISIÓN PRIMARIA DE ACCIONES", "level": 2, "is_total": False, "is_detail": True},
        {"column": "RESERVAS", "label": "RESERVAS", "level": 2, "is_total": False, "is_detail": True},
        {"column": "OTROS RESULTADOS INTEGRALES", "label": "OTROS RESULTADOS INTEGRALES", "level": 2, "is_total": False, "is_detail": True},
        {"column": "RESULTADOS ACUMULADOS", "label": "RESULTADOS ACUMULADOS", "level": 2, "is_total": False, "is_detail": True},
        {"column": "RESULTADOS DEL EJERCICIO", "label": "RESULTADOS DEL EJERCICIO", "level": 2, "is_total": False, "is_detail": True},
    ]

    balance_columns = [item["column"] for item in balance_structure]
    balance_company_df = balance_data[balance_data["RUC"] == str(ruc)].copy()
    if balance_company_df.empty:
        balance_company_df = balance_data[balance_data["NOMBRE"] == selected_company].copy()

    if balance_company_df.empty:
        st.warning("No se encontro informacion de balance general para esta empresa.")
    else:
        available_balance_cols = [col for col in balance_columns if col in balance_company_df.columns]
        missing_balance_cols = [col for col in balance_columns if col not in balance_company_df.columns]
        annual_balance_df = (
            balance_company_df.groupby("AÑO", dropna=False)[available_balance_cols]
            .sum(numeric_only=True)
            .reindex([2023, 2024])
        )

        vertical_parent = {item["column"]: "ACTIVO" for item in balance_structure if "ACTIVO" in item["column"] and item["column"] != "PATRIMONIO"}
        vertical_parent.update({item["column"]: "PASIVO" for item in balance_structure if "PASIVO" in item["column"]})
        vertical_parent.update(
            {
                "ACTIVO": "ACTIVO",
                "PASIVO": "PASIVO",
                "PATRIMONIO": "PATRIMONIO",
                "CAPITAL": "PATRIMONIO",
                "APORTES PARA FUTURA CAPITALIZACIÓN": "PATRIMONIO",
                "PRIMA POR EMISIÓN PRIMARIA DE ACCIONES": "PATRIMONIO",
                "RESERVAS": "PATRIMONIO",
                "OTROS RESULTADOS INTEGRALES": "PATRIMONIO",
                "RESULTADOS ACUMULADOS": "PATRIMONIO",
                "RESULTADOS DEL EJERCICIO": "PATRIMONIO",
            }
        )

        balance_rows = []
        rows_by_col = {}
        for item in balance_structure:
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

            row = {
                "_column": col,
                "_label": item["label"],
                "_level": item.get("level", 0),
                "_is_total": item.get("is_total", False),
                "Cuenta": item["label"],
                "2023": value_2023,
                "2024": value_2024,
                "Variacion 2024/2023": var_abs,
                "Comparativo Horizontal": var_pct,
                "Comparativo Vertical": vertical_pct,
            }
            balance_rows.append(row)
            rows_by_col[col] = row

        balance_df = pd.DataFrame(balance_rows)
        if missing_balance_cols:
            st.warning(f"Columnas de balance no encontradas: {', '.join(missing_balance_cols)}")

        children_map = {
            "ACTIVO": ["ACTIVO CORRIENTE", "ACTIVO NO CORRIENTE"],
            "ACTIVO CORRIENTE": [
                "DISPONIBLE",
                "INVERSIONES",
                "CUENTAS POR COBRAR CLIENTES",
                "CUENTAS POR COBRAR RELACIONADAS",
                "PROVISIÓN INCOBRABLE Y DETERIORO",
                "INVENTARIO",
                "CRÉDITO TRIBUTARIO",
                "OTROS ACTIVOS CORRIENTES",
            ],
            "ACTIVO NO CORRIENTE": [
                "PROPIEDAD, PLANTA Y EQUIPO",
                "PROPIEDADES DE INVERSIÓN",
                "ACTIVOS BIOLÓGICOS",
                "ACTIVO INTANGIBLE",
                "ACTIVOS POR IMPUESTOS DIFERIDOS",
                "ACTIVOS FINANCIEROS NO CORRIENTES",
                "DERECHO DE USO POR ACTIVOS ARRENDADOS",
                "DOCUMENTOS Y CUENTAS POR COBRAR NO RELACIONADOS",
                "DOCUMENTOS Y CUENTAS POR COBRAR RELACIONADOS",
                "OTROS ACTIVOS NO CORRIENTES",
            ],
            "PASIVO": ["PASIVO CORRIENTE", "PASIVO NO CORRIENTE"],
            "PASIVO CORRIENTE": [
                "CUENTAS POR PAGAR",
                "OBLIGACIONES FINACIERAS CORTO PLAZO",
                "IMPUESTOS POR PAGAR",
                "OTRAS CUENTAS POR PAGAR",
                "PROVISIONES",
                "OTROS PASIVOS CORRIENTES",
            ],
            "PASIVO NO CORRIENTE": [
                "OBLIGACIONES FINANCIERAS LARGO PLAZO",
                "ARRENDAMIENTO LARGO PLAZO",
                "PASIVO DIFERIDO",
                "BENEFICIOS EMPLEADOS LARGO PLAZO",
                "OTRAS CUENTAS POR PAGAR LARGO PLAZO",
                "OTROS PASIVOS NO CORRIENTES",
            ],
            "PATRIMONIO": [
                "CAPITAL",
                "APORTES PARA FUTURA CAPITALIZACIÓN",
                "PRIMA POR EMISIÓN PRIMARIA DE ACCIONES",
                "RESERVAS",
                "OTROS RESULTADOS INTEGRALES",
                "RESULTADOS ACUMULADOS",
                "RESULTADOS DEL EJERCICIO",
            ],
        }

        show_level_1 = st.checkbox("Desagregar cuentas Nivel 1", value=False, key=f"bal_l1_{ruc}")
        show_level_2 = st.checkbox("Desagregar cuentas Nivel 2", value=False, key=f"bal_l2_{ruc}")
        show_level_1_effective = show_level_1 or show_level_2

        visible_rows = []

        def append_node(node: str, depth: int) -> None:
            row = rows_by_col.get(node)
            if row is None:
                return

            row_copy = row.copy()
            if depth == 0:
                row_copy["Cuenta"] = f"= {row_copy['_label']}"
            elif depth == 1:
                row_copy["Cuenta"] = f"   |- {row_copy['_label']}"
            else:
                row_copy["Cuenta"] = f"      |--- {row_copy['_label']}"

            visible_rows.append(row_copy)
            if depth == 0 and show_level_1_effective:
                for child in children_map.get(node, []):
                    append_node(child, depth + 1)
            elif depth == 1 and show_level_2:
                for child in children_map.get(node, []):
                    append_node(child, depth + 1)

        for root in ["ACTIVO", "PASIVO", "PATRIMONIO"]:
            append_node(root, 0)

        visible_df = pd.DataFrame(visible_rows)
        visible_total_rows = set(visible_df.index[visible_df["_is_total"]].tolist())
        visible_level_map = {idx: int(level) for idx, level in visible_df["_level"].items()}
        view_df = visible_df.drop(columns=["_column", "_label", "_level", "_is_total"])

        styled_balance = (
            view_df.style
            .format(
                {
                    "2023": "{:,.0f}",
                    "2024": "{:,.0f}",
                    "Variacion 2024/2023": "{:,.0f}",
                    "Comparativo Horizontal": "{:.2f}%",
                    "Comparativo Vertical": "{:.2f}%",
                },
                na_rep="-",
            )
            .apply(lambda row: style_balance_row(row, visible_total_rows, visible_level_map), axis=1)
            .apply(lambda row: style_hierarchy_label(row, visible_level_map), axis=1, subset=["Cuenta"])
        )

        st.dataframe(styled_balance, width="stretch", hide_index=True)
        balance_export_df = view_df.copy()
        try:
            balance_bytes = to_excel_bytes(balance_export_df, "Balance_General")
            balance_file = f"analisis_balance_general_{safe_filename(selected_company)}.xlsx"
            st.download_button(
                "Descargar Excel - Balance General",
                data=balance_bytes,
                file_name=balance_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_bg_{safe_filename(selected_company)}",
            )
        except RuntimeError:
            st.error("No se pudo generar Excel para Balance General (falta xlsxwriter/openpyxl).")
else:
    st.info("Selecciona una empresa para visualizar su analisis financiero.")
