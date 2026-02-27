import base64
import io
from pathlib import Path

import altair as alt
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
def load_financial_data(path_2021_2022: str, path_2023_2024: str) -> pd.DataFrame:
    df_2021_2022 = safe_read_pickle(path_2021_2022)
    df_2023_2024 = safe_read_pickle(path_2023_2024)
    df = pd.concat([df_2021_2022, df_2023_2024], ignore_index=True)
    df["AÑO"] = pd.to_numeric(df["AÑO"], errors="coerce").astype("Int64")
    df["NOMBRE"] = df["NOMBRE"].astype(str).str.strip()
    return df


@st.cache_data
def load_balance_data(path_2022: str, path_2023: str, path_2024: str) -> pd.DataFrame:
    df_2022 = safe_read_pickle(path_2022)
    df_2023 = safe_read_pickle(path_2023)
    df_2024 = safe_read_pickle(path_2024)

    df_2022["AÑO"] = 2022
    df_2023["AÑO"] = 2023
    df_2024["AÑO"] = 2024

    for df in [df_2022, df_2023, df_2024]:
        df["NOMBRE"] = df["NOMBRE"].astype(str).str.strip()
        df["RUC"] = df["RUC"].astype(str).str.strip()

    return pd.concat([df_2022, df_2023, df_2024], ignore_index=True)


@st.cache_data
def load_indicators_data(path: str) -> pd.DataFrame:
    df = safe_read_pickle(path)
    if "AÑO" in df.columns:
        df["AÑO"] = pd.to_numeric(df["AÑO"], errors="coerce").astype("Int64")
    if "NOMBRE" in df.columns:
        df["NOMBRE"] = df["NOMBRE"].astype(str).str.strip()
    if "RUC" in df.columns:
        df["RUC"] = df["RUC"].astype(str).str.strip()
    return df


@st.cache_data
def load_company_directory_data(path: str) -> pd.DataFrame:
    df = safe_read_pickle(path)
    if "RUC" in df.columns:
        df["RUC"] = df["RUC"].astype(str).str.strip()
    return df


@st.cache_data
def load_ciiu_catalog(path: str) -> pd.DataFrame:
    df = pd.read_excel(path).copy()
    if "Codigo" not in df.columns or "Descripcion" not in df.columns:
        # Fallback defensivo por si cambia el encabezado del archivo.
        base_cols = df.columns.tolist()
        if len(base_cols) >= 2:
            df = df.rename(columns={base_cols[0]: "Codigo", base_cols[1]: "Descripcion"})
    df = df.dropna(subset=["Codigo", "Descripcion"]).copy()
    df["Codigo"] = df["Codigo"].astype(str).str.strip().str.upper()
    df["Codigo_flat"] = df["Codigo"].str.replace(r"[^A-Z0-9]", "", regex=True)
    df["Descripcion"] = df["Descripcion"].astype(str).str.strip()
    return df


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


def escape_excel_formula_value(value):
    if isinstance(value, str) and value and value[0] in ("=", "+", "-", "@"):
        return "'" + value
    return value


def sanitize_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    safe_df = df.copy()
    text_cols = safe_df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        safe_df[col] = safe_df[col].map(escape_excel_formula_value)
    return safe_df


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    safe_df = sanitize_for_excel(df)
    for engine in ("xlsxwriter", "openpyxl"):
        try:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine=engine) as writer:
                safe_df.to_excel(writer, index=False, sheet_name=sheet_name)
            buffer.seek(0)
            return buffer.getvalue()
        except Exception:
            continue

    raise RuntimeError("No hay motor de Excel disponible (xlsxwriter/openpyxl).")


def safe_filename(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(text)).strip("_")


def strip_simple_html(text: str) -> str:
    return str(text).replace("<strong>", "").replace("</strong>", "")


def to_indicators_pdf_bytes(
    company_name: str,
    ruc: str,
    total_score: float | None,
    status_label: str,
    narrative_lines: list[str],
    block_scores: dict[str, float | None],
    indicators_table_df: pd.DataFrame,
) -> bytes:
    try:
        from fpdf import FPDF
    except Exception as exc:
        raise RuntimeError("No hay motor PDF disponible (falta fpdf2).") from exc

    def pdf_text(value) -> str:
        text = strip_simple_html(value if value is not None else "")
        text = str(text).replace("\n", " ").strip()
        return text.encode("latin-1", errors="replace").decode("latin-1")

    class IndicatorsPDF(FPDF):
        def footer(self) -> None:
            self.set_y(-10)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(107, 114, 128)
            self.cell(0, 5, f"Página {self.page_no()}", align="R")

    pdf = IndicatorsPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.set_title(pdf_text(f"Indicadores Financieros Clave - {company_name}"))
    pdf.set_author("Andersen Intelligence")
    pdf.add_page()

    page_width = 190
    table_col_widths = [94, 24, 24, 24, 24]
    row_h = 7

    # Header card
    pdf.set_draw_color(199, 210, 254)
    pdf.set_fill_color(248, 250, 255)
    card_h = 38
    pdf.rect(10, 10, page_width, card_h, "DF")
    has_logo = False
    logo_path = Path("Logo Andersen.png")
    if logo_path.exists():
        try:
            # Bigger logo, constrained inside the header card.
            pdf.image(str(logo_path), x=154, y=14, w=40, h=18)
            has_logo = True
        except Exception:
            has_logo = False

    info_w = 136 if has_logo else 0
    pdf.set_xy(14, 13)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(17, 24, 39)
    pdf.cell(info_w, 6, pdf_text("Indicadores Financieros Clave"), ln=1)
    pdf.set_x(14)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(info_w, 5, pdf_text(f"Empresa: {company_name}"), ln=1)
    pdf.set_x(14)
    pdf.cell(info_w, 5, pdf_text(f"RUC: {ruc}"), ln=1)
    pdf.set_x(14)
    if total_score is not None:
        score_line = f"Score total: {total_score:.1f}/100 | Estado: {status_label.capitalize()}"
    else:
        score_line = "Score total: N/D | Estado: Sin datos"
    pdf.cell(info_w, 5, pdf_text(score_line), ln=1)
    # Ensure the executive summary starts below the blue header card.
    pdf.set_y(10 + card_h + 6)

    # Narrative
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(17, 24, 39)
    pdf.cell(0, 6, pdf_text("Resumen Ejecutivo"), ln=1)
    pdf.set_font("Helvetica", "", 9.8)
    pdf.set_text_color(31, 41, 55)
    for line in narrative_lines:
        pdf.multi_cell(0, 5.2, pdf_text(line))
        pdf.ln(0.5)

    # Score table
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, pdf_text("Scores por Bloque"), ln=1)
    pdf.set_fill_color(31, 41, 55)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(55, row_h, "Bloque", border=1, fill=True)
    pdf.cell(35, row_h, "Score", border=1, fill=True, align="R", ln=1)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(31, 41, 55)
    for block in ["Rentabilidad", "Estructura", "Liquidez"]:
        value = block_scores.get(block)
        score_text = f"{value:.1f}/100" if value is not None else "N/D"
        pdf.cell(55, row_h, pdf_text(block), border=1)
        pdf.cell(35, row_h, pdf_text(score_text), border=1, align="R", ln=1)

    # Indicators table
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, pdf_text("Tabla de Indicadores (2021-2024)"), ln=1)

    def write_table_header() -> None:
        pdf.set_fill_color(31, 41, 55)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8.6)
        headers = ["Indicador", "2021", "2022", "2023", "2024"]
        for idx, header in enumerate(headers):
            align = "L" if idx == 0 else "R"
            end_ln = 1 if idx == len(headers) - 1 else 0
            pdf.cell(table_col_widths[idx], row_h, pdf_text(header), border=1, fill=True, align=align, ln=end_ln)
        pdf.set_text_color(31, 41, 55)
        pdf.set_font("Helvetica", "", 8.4)

    write_table_header()

    for _, row in indicators_table_df.iterrows():
        indicator = pdf_text(row.get("Indicador", "-"))
        y2021 = pdf_text(row.get("2021", "-"))
        y2022 = pdf_text(row.get("2022", "-"))
        y2023 = pdf_text(row.get("2023", "-"))
        y2024 = pdf_text(row.get("2024", "-"))

        if pdf.get_y() > 272:
            pdf.add_page()
            write_table_header()

        if indicator.upper().startswith("INDICADORES DE "):
            pdf.set_fill_color(238, 242, 255)
            pdf.set_font("Helvetica", "B", 8.6)
            pdf.cell(sum(table_col_widths), row_h, indicator, border=1, fill=True, ln=1)
            pdf.set_font("Helvetica", "", 8.4)
            continue

        indicator_short = indicator[:60]
        pdf.cell(table_col_widths[0], row_h, indicator_short, border=1, align="L")
        pdf.cell(table_col_widths[1], row_h, y2021[:16], border=1, align="R")
        pdf.cell(table_col_widths[2], row_h, y2022[:16], border=1, align="R")
        pdf.cell(table_col_widths[3], row_h, y2023[:16], border=1, align="R")
        pdf.cell(table_col_widths[4], row_h, y2024[:16], border=1, align="R", ln=1)

    output = pdf.output(dest="S")
    if isinstance(output, str):
        return output.encode("latin-1", errors="replace")
    return bytes(output)


def safe_read_pickle(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    head = file_path.read_bytes()[:256]
    if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"El archivo {path} es un puntero de Git LFS, no un pickle real. "
            "En Streamlit Cloud debes subir el archivo binario real o evitar LFS para ese archivo."
        )

    try:
        return pd.read_pickle(path).copy()
    except Exception as exc:
        header_hint = ""
        if not head.startswith(b"\x80"):
            header_text = head.decode("utf-8", errors="replace").strip()
            if header_text:
                header_hint = f" Encabezado detectado: {header_text[:100]!r}."
        raise RuntimeError(
            f"No se pudo leer {path} como pickle ({type(exc).__name__}).{header_hint}"
        ) from exc


def score_high_is_better(value, good_threshold: float, neutral_threshold: float) -> float | None:
    if pd.isna(value):
        return None
    if value >= good_threshold:
        return 100.0
    if value >= neutral_threshold:
        return 65.0
    return 30.0


def score_low_is_better(value, good_threshold: float, neutral_threshold: float) -> float | None:
    if pd.isna(value):
        return None
    if value <= good_threshold:
        return 100.0
    if value <= neutral_threshold:
        return 65.0
    return 30.0


def trend_adjustment(value_start, value_end, higher_is_better: bool, tolerance: float) -> float:
    if pd.isna(value_start) or pd.isna(value_end):
        return 0.0
    delta = float(value_end) - float(value_start)
    if abs(delta) <= tolerance:
        return 0.0
    if higher_is_better:
        return 10.0 if delta > 0 else -10.0
    return 10.0 if delta < 0 else -10.0


def clamp_score(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(100.0, value))


def average_score(scores: list[float | None]) -> float | None:
    valid_scores = [float(s) for s in scores if s is not None]
    if not valid_scores:
        return None
    return sum(valid_scores) / len(valid_scores)


def score_label(score: float | None) -> str:
    if score is None:
        return "sin datos"
    if score >= 75:
        return "fuerte"
    if score >= 60:
        return "estable"
    if score >= 45:
        return "en observación"
    return "débil"


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

data = None
balance_data = None
critical_load_errors = []
try:
    data = load_financial_data("supercias_resultados_2021_2022.pkl", "supercias_resultados_2023_2024.pkl")
except Exception as exc:
    critical_load_errors.append(str(exc))
try:
    balance_data = load_balance_data("supercias_balances_2022.pkl", "supercias_balances_2023.pkl", "supercias_balances_2024.pkl")
except Exception as exc:
    critical_load_errors.append(str(exc))

if data is None or balance_data is None:
    st.error("No se pudieron cargar los datasets principales de resultados/balance.")
    for err in critical_load_errors:
        st.code(err)
    st.info(
        "Si usas Streamlit Cloud, revisa que los .pkl sean binarios reales y no punteros de Git LFS. "
        "También valida que los archivos estén completos en el repositorio."
    )
    st.stop()

indicators_data = None
try:
    indicators_data = load_indicators_data("supercias_indicadores.pkl")
except Exception:
    indicators_data = None
company_directory_data = None
try:
    company_directory_data = load_company_directory_data("directorio_core.pickle")
except Exception:
    company_directory_data = None
ciiu_catalog_data = None
try:
    ciiu_catalog_data = load_ciiu_catalog("CIIU.xlsx")
except Exception:
    ciiu_catalog_data = None

statement_structure = [
    {"column": "INGRESOS", "label": "INGRESOS", "sign": "", "is_total": False, "is_detail": False},
    {"column": "COSTO DE VENTAS", "label": "COSTO DE VENTAS", "sign": "(-)", "is_total": False, "is_detail": False},
    {"column": "COSTO DE DISTRIBUCIÓN", "label": "COSTO DE DISTRIBUCIÓN", "sign": "(-)", "is_total": False, "is_detail": False},
    {"column": "CONTRIBUCIÓN MARGINAL", "label": "CONTRIBUCIÓN MARGINAL", "sign": "=", "is_total": True, "is_detail": False},
    {"column": "GASTO OPERACIONAL", "label": "GASTO OPERACIONAL", "sign": "(-)", "is_total": False, "is_detail": False},
    {"column": "GASTO DE PERSONAL", "label": "GASTO DE PERSONAL", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO DE ARRENDAMIENTO", "label": "GASTO DE ARRENDAMIENTO", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO DE PUBLICIDAD", "label": "GASTO DE PUBLICIDAD", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO DE SERVICIOS BASICOS", "label": "GASTO DE SERVICIOS BASICOS", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO DE GESTION", "label": "GASTO DE GESTION", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO DE MANTENIMIENTO", "label": "GASTO DE MANTENIMIENTO", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO DE SERVICIOS PROFESIONALES", "label": "GASTO DE SERVICIOS PROFESIONALES", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO DE SEGUROS", "label": "GASTO DE SEGUROS", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "OTROS GASTOS", "label": "OTROS GASTOS", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "UTILIDAD OPERACIONAL", "label": "UTILIDAD OPERACIONAL", "sign": "=", "is_total": True, "is_detail": False},
    {"column": "GASTOS ADMINISTRATIVOS", "label": "GASTOS ADMINISTRATIVOS", "sign": "(-)", "is_total": False, "is_detail": False},
    {"column": "GASTO DE PERSONAL ADMINISTRATIVO", "label": "GASTO DE PERSONAL ADMINISTRATIVO", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO DE SERVICIOS PROFESIONALES ADMINISTRATIVOS", "label": "GASTO DE SERVICIOS PROFESIONALES ADMINISTRATIVOS", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO DE ARRENDAMIENTO ADMINISTRATIVO", "label": "GASTO DE ARRENDAMIENTO ADMINISTRATIVO", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "IMPUESTOS, TASAS Y CONTRIBUCIONES", "label": "IMPUESTOS, TASAS Y CONTRIBUCIONES", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO DE SERVICIOS BASICOS ADMINISTRATIVOS", "label": "GASTO DE SERVICIOS BASICOS ADMINISTRATIVOS", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO DE MANTENIMIENTO ADMINISTRATIVO", "label": "GASTO DE MANTENIMIENTO ADMINISTRATIVO", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO DE TRANSPORTE Y GESTIÓN ADMINISTRATIVO", "label": "GASTO DE TRANSPORTE Y GESTIÓN ADMINISTRATIVO", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO DE SEGUROS ADMINISTRATIVOS", "label": "GASTO DE SEGUROS ADMINISTRATIVOS", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "GASTO DE PUBLICIDAD ADMINISTRATIVO", "label": "GASTO DE PUBLICIDAD ADMINISTRATIVO", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "OTROS GASTOS ADMINISTRATIVOS", "label": "OTROS GASTOS ADMINISTRATIVOS", "sign": "->", "is_total": False, "is_detail": True},
    {"column": "EBITDA", "label": "EBITDA", "sign": "=", "is_total": True, "is_detail": False},
    {"column": "DEPRECIACION", "label": "DEPRECIACION", "sign": "(+)", "is_total": False, "is_detail": False},
    {"column": "AMORTIZACION", "label": "AMORTIZACION", "sign": "(+)", "is_total": False, "is_detail": False},
    {"column": "COSTO FINANCIERO", "label": "COSTO FINANCIERO", "sign": "(-)", "is_total": False, "is_detail": False},
    {"column": "RESULTADO ANTES DE IMPUESTOS", "label": "RESULTADO ANTES DE IMPUESTOS", "sign": "=", "is_total": True, "is_detail": False},
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
        "CONTRIBUCIÓN MARGINAL",
        "GASTO OPERACIONAL",
        "UTILIDAD OPERACIONAL",
        "GASTOS ADMINISTRATIVOS",
        "EBITDA",
        "RESULTADO ANTES DE IMPUESTOS",
    }

    annual_df = (
        company_df.groupby("AÑO", dropna=False)[available_columns]
        .sum(numeric_only=True)
        .reindex([2021, 2022, 2023, 2024])
    )

    ruc = company_df["RUC"].dropna().astype(str).iloc[0] if not company_df["RUC"].dropna().empty else "-"
    st.subheader(selected_company)
    st.write(f"**RUC:** {ruc}")
    tab_profile, tab_pyg, tab_bg, tab_ind, tab_graph = st.tabs(
        [
            "Perfil de la Compañía",
            "Analisis de Perdidas y Ganancias",
            "Analisis de Balance General",
            "Indicadores Financieros Clave",
            "Gráficos Seleccionados",
        ]
    )

    with tab_profile:
        st.markdown("#### Perfil de la Compañía")
        st.write(f"**Nombre:** {selected_company}")
        st.write(f"**RUC:** {ruc}")
        profile_fields = [
            ("Fecha de Constitución", "FECHA_CONSTITUCION"),
            ("Capital Suscrito", "CAPITAL SUSCRITO"),
            ("Provincia", "PROVINCIA"),
            ("Cantón", "CANTÓN"),
            ("Tipo", "TIPO"),
            ("Situación Legal", "SITUACIÓN LEGAL"),
            ("Representante", "REPRESENTANTE"),
            ("Cargo", "CARGO"),
        ]

        if company_directory_data is None:
            st.info("No se encontró el archivo directorio_core.pickle para mostrar más datos de perfil.")
        else:
            company_profile_df = company_directory_data[company_directory_data["RUC"] == str(ruc)].copy()
            if company_profile_df.empty:
                st.info("No se encontró información adicional de perfil para esta empresa.")
            else:
                profile_row = company_profile_df.iloc[0]
                left_col, right_col = st.columns(2)
                for idx, (label, column) in enumerate(profile_fields):
                    value = profile_row[column] if column in company_profile_df.columns else pd.NA
                    if pd.isna(value):
                        display_value = "-"
                    elif column == "CAPITAL SUSCRITO":
                        display_value = f"{float(value):,.2f}"
                    else:
                        display_value = str(value)

                    target_col = left_col if idx % 2 == 0 else right_col
                    target_col.write(f"**{label}:** {display_value}")

                ciiu_code = (
                    str(profile_row["CIIU NIVEL 6"]).strip().upper()
                    if "CIIU NIVEL 6" in company_profile_df.columns and pd.notna(profile_row["CIIU NIVEL 6"])
                    else ""
                )
                ciiu_description = "-"
                if ciiu_catalog_data is not None and ciiu_code:
                    ciiu_match = ciiu_catalog_data[ciiu_catalog_data["Codigo"] == ciiu_code]
                    if ciiu_match.empty:
                        ciiu_code_flat = "".join(ch for ch in ciiu_code if ch.isalnum())
                        ciiu_match = ciiu_catalog_data[ciiu_catalog_data["Codigo_flat"] == ciiu_code_flat]
                    if not ciiu_match.empty:
                        ciiu_description = str(ciiu_match.iloc[0]["Descripcion"])

                st.write(f"**CIIU Nivel 6:** {ciiu_code if ciiu_code else '-'}")
                st.write(f"**Actividad Económica:** {ciiu_description}")

    with tab_pyg:
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

    with tab_bg:
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
            {"column": "OBLIGACIONES FINANCIERAS CORTO PLAZO", "label": "OBLIGACIONES FINANCIERAS CORTO PLAZO", "level": 2, "is_total": False, "is_detail": True},
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
                .reindex([2022, 2023, 2024])
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
                value_2022 = annual_balance_df.loc[2022, col] if col in annual_balance_df.columns else pd.NA
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
                    "2022": value_2022,
                    "2023": value_2023,
                    "2024": value_2024,
                    "Variacion 2024/2023": var_abs,
                    "Comparativo Horizontal": var_pct,
                    "Comparativo Vertical": vertical_pct,
                }
                balance_rows.append(row)
                rows_by_col[col] = row

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
                    "OBLIGACIONES FINANCIERAS CORTO PLAZO",
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
                        "2022": "{:,.0f}",
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

    with tab_ind:
        st.markdown("#### Indicadores Financieros Clave")
        indicator_list = [
            "MARGEN BRUTO",
            "MARGEN EBITDA",
            "MARGEN DE UTILIDAD",
            "ROA",
            "ROE",
            "ENDEUDAMIENTO",
            "DÍAS DE INVENTARIO",
            "DÍAS DE COBRO",
            "DÍAS DE PAGO",
            "CICLO DE CONVERSIÓN DE EFECTIVO",
            "RAZÓN CORRIENTE",
            "PRUEBA ÁCIDA",
        ]

        if indicators_data is None:
            st.warning("No se encontro el archivo supercias_indicadores.pkl para mostrar esta seccion.")
        else:
            indicators_company_df = pd.DataFrame()
            if "RUC" in indicators_data.columns:
                indicators_company_df = indicators_data[indicators_data["RUC"] == str(ruc)].copy()
            if indicators_company_df.empty and "NOMBRE" in indicators_data.columns:
                indicators_company_df = indicators_data[indicators_data["NOMBRE"] == selected_company].copy()

            if indicators_company_df.empty:
                st.warning("No se encontro informacion de indicadores para esta empresa.")
            elif "AÑO" not in indicators_company_df.columns:
                st.warning("El archivo de indicadores no contiene la columna AÑO.")
            else:
                available_indicators = [col for col in indicator_list if col in indicators_company_df.columns]
                missing_indicators = [col for col in indicator_list if col not in indicators_company_df.columns]
                annual_indicators_df = (
                    indicators_company_df.groupby("AÑO", dropna=False)[available_indicators]
                    .mean(numeric_only=True)
                    .reindex([2021, 2022, 2023, 2024])
                )

                def get_indicator_value(year: int, column: str):
                    if column in annual_indicators_df.columns and year in annual_indicators_df.index:
                        return annual_indicators_df.loc[year, column]
                    return pd.NA

                # Resumen automatico: reglas objetivas con umbrales y tendencia 2022->2024
                ingresos_2023 = annual_df.loc[2023, "INGRESOS"] if "INGRESOS" in annual_df.columns else pd.NA
                ingresos_2024 = annual_df.loc[2024, "INGRESOS"] if "INGRESOS" in annual_df.columns else pd.NA
                crecimiento_ingresos = pd.NA
                if pd.notna(ingresos_2023) and pd.notna(ingresos_2024) and ingresos_2023 != 0:
                    crecimiento_ingresos = (ingresos_2024 - ingresos_2023) / abs(ingresos_2023)

                margen_ebitda_2024 = get_indicator_value(2024, "MARGEN EBITDA")
                margen_ebitda_2022 = get_indicator_value(2022, "MARGEN EBITDA")
                margen_utilidad_2024 = get_indicator_value(2024, "MARGEN DE UTILIDAD")
                margen_utilidad_2022 = get_indicator_value(2022, "MARGEN DE UTILIDAD")
                roa_2024 = get_indicator_value(2024, "ROA")
                roa_2022 = get_indicator_value(2022, "ROA")
                roe_2024 = get_indicator_value(2024, "ROE")
                roe_2022 = get_indicator_value(2022, "ROE")
                endeudamiento_2024 = get_indicator_value(2024, "ENDEUDAMIENTO")
                endeudamiento_2022 = get_indicator_value(2022, "ENDEUDAMIENTO")
                razon_corriente_2024 = get_indicator_value(2024, "RAZÓN CORRIENTE")
                razon_corriente_2022 = get_indicator_value(2022, "RAZÓN CORRIENTE")
                prueba_acida_2024 = get_indicator_value(2024, "PRUEBA ÁCIDA")
                prueba_acida_2022 = get_indicator_value(2022, "PRUEBA ÁCIDA")
                ccc_2024 = get_indicator_value(2024, "CICLO DE CONVERSIÓN DE EFECTIVO")
                ccc_2022 = get_indicator_value(2022, "CICLO DE CONVERSIÓN DE EFECTIVO")

                patrimonio_activo_2024 = pd.NA
                patrimonio_activo_2022 = pd.NA
                balance_company_df_summary = balance_data[balance_data["RUC"] == str(ruc)].copy()
                if balance_company_df_summary.empty:
                    balance_company_df_summary = balance_data[balance_data["NOMBRE"] == selected_company].copy()
                if (
                    not balance_company_df_summary.empty
                    and "ACTIVO" in balance_company_df_summary.columns
                    and "PATRIMONIO" in balance_company_df_summary.columns
                ):
                    annual_balance_summary = (
                        balance_company_df_summary.groupby("AÑO", dropna=False)[["ACTIVO", "PATRIMONIO"]]
                        .sum(numeric_only=True)
                        .reindex([2022, 2024])
                    )
                    activo_2024 = annual_balance_summary.loc[2024, "ACTIVO"]
                    patrimonio_2024 = annual_balance_summary.loc[2024, "PATRIMONIO"]
                    activo_2022 = annual_balance_summary.loc[2022, "ACTIVO"]
                    patrimonio_2022 = annual_balance_summary.loc[2022, "PATRIMONIO"]
                    if pd.notna(activo_2024) and pd.notna(patrimonio_2024) and activo_2024 > 0:
                        patrimonio_activo_2024 = patrimonio_2024 / activo_2024
                    if pd.notna(activo_2022) and pd.notna(patrimonio_2022) and activo_2022 > 0:
                        patrimonio_activo_2022 = patrimonio_2022 / activo_2022

                summary_metrics = {"Rentabilidad": [], "Estructura": [], "Liquidez": []}

                def add_metric(block: str, name: str, value_2024, base_score: float | None, trend_score: float = 0.0, value_type: str = "ratio"):
                    final_score = clamp_score(base_score + trend_score) if base_score is not None else None
                    if pd.isna(value_2024):
                        value_text = "N/D"
                    elif value_type == "percent":
                        value_text = f"{float(value_2024) * 100:.2f}%"
                    elif value_type == "days":
                        value_text = f"{float(value_2024):,.1f} días"
                    else:
                        value_text = f"{float(value_2024):,.2f}"
                    summary_metrics[block].append(
                        {"name": name, "score": final_score, "value_text": value_text, "trend_score": trend_score}
                    )

                add_metric(
                    "Rentabilidad",
                    "Crecimiento de ingresos",
                    crecimiento_ingresos,
                    score_high_is_better(crecimiento_ingresos, 0.10, 0.00),
                    value_type="percent",
                )
                add_metric(
                    "Rentabilidad",
                    "Margen EBITDA",
                    margen_ebitda_2024,
                    score_high_is_better(margen_ebitda_2024, 0.15, 0.08),
                    trend_adjustment(margen_ebitda_2022, margen_ebitda_2024, True, 0.005),
                    "percent",
                )
                add_metric(
                    "Rentabilidad",
                    "Margen de utilidad",
                    margen_utilidad_2024,
                    score_high_is_better(margen_utilidad_2024, 0.08, 0.03),
                    trend_adjustment(margen_utilidad_2022, margen_utilidad_2024, True, 0.005),
                    "percent",
                )
                add_metric(
                    "Rentabilidad",
                    "ROA",
                    roa_2024,
                    score_high_is_better(roa_2024, 0.08, 0.04),
                    trend_adjustment(roa_2022, roa_2024, True, 0.005),
                    "percent",
                )
                add_metric(
                    "Rentabilidad",
                    "ROE",
                    roe_2024,
                    score_high_is_better(roe_2024, 0.15, 0.08),
                    trend_adjustment(roe_2022, roe_2024, True, 0.005),
                    "percent",
                )
                add_metric(
                    "Estructura",
                    "Endeudamiento",
                    endeudamiento_2024,
                    score_low_is_better(endeudamiento_2024, 0.50, 0.70),
                    trend_adjustment(endeudamiento_2022, endeudamiento_2024, False, 0.01),
                    "percent",
                )
                add_metric(
                    "Estructura",
                    "Patrimonio/Activo",
                    patrimonio_activo_2024,
                    score_high_is_better(patrimonio_activo_2024, 0.40, 0.25),
                    trend_adjustment(patrimonio_activo_2022, patrimonio_activo_2024, True, 0.01),
                    "percent",
                )
                add_metric(
                    "Liquidez",
                    "Razón corriente",
                    razon_corriente_2024,
                    score_high_is_better(razon_corriente_2024, 1.50, 1.10),
                    trend_adjustment(razon_corriente_2022, razon_corriente_2024, True, 0.05),
                    "ratio",
                )
                add_metric(
                    "Liquidez",
                    "Prueba ácida",
                    prueba_acida_2024,
                    score_high_is_better(prueba_acida_2024, 1.00, 0.70),
                    trend_adjustment(prueba_acida_2022, prueba_acida_2024, True, 0.05),
                    "ratio",
                )
                add_metric(
                    "Liquidez",
                    "Ciclo de conversión de efectivo",
                    ccc_2024,
                    score_low_is_better(ccc_2024, 30.0, 60.0),
                    trend_adjustment(ccc_2022, ccc_2024, False, 2.0),
                    "days",
                )

                block_scores = {
                    "Rentabilidad": average_score([item["score"] for item in summary_metrics["Rentabilidad"]]),
                    "Estructura": average_score([item["score"] for item in summary_metrics["Estructura"]]),
                    "Liquidez": average_score([item["score"] for item in summary_metrics["Liquidez"]]),
                }
                block_trend_scores = {
                    "Rentabilidad": average_score([item["trend_score"] for item in summary_metrics["Rentabilidad"]]),
                    "Estructura": average_score([item["trend_score"] for item in summary_metrics["Estructura"]]),
                    "Liquidez": average_score([item["trend_score"] for item in summary_metrics["Liquidez"]]),
                }
                block_weights = {"Rentabilidad": 0.40, "Estructura": 0.30, "Liquidez": 0.30}
                valid_blocks = {k: v for k, v in block_scores.items() if v is not None}
                pdf_total_score = None
                pdf_status_label = "sin datos"
                pdf_narratives = ["No hay datos suficientes para construir el resumen automático de indicadores."]
                pdf_block_scores = block_scores.copy()

                if valid_blocks:
                    valid_weight_sum = sum(block_weights[k] for k in valid_blocks)
                    total_score = sum(valid_blocks[k] * block_weights[k] for k in valid_blocks) / valid_weight_sum

                    best_block = max(valid_blocks, key=valid_blocks.get)
                    worst_block = min(valid_blocks, key=valid_blocks.get)
                    best_metric = max(
                        [m for m in summary_metrics[best_block] if m["score"] is not None],
                        key=lambda m: m["score"],
                    )
                    worst_metric = min(
                        [m for m in summary_metrics[worst_block] if m["score"] is not None],
                        key=lambda m: m["score"],
                    )

                    block_names = {
                        "Rentabilidad": "rentabilidad",
                        "Estructura": "estructura financiera",
                        "Liquidez": "liquidez",
                    }
                    priority_map = {
                        "Rentabilidad": "Prioridad: reforzar margen operativo y disciplina de costos para sostener la rentabilidad.",
                        "Estructura": "Prioridad: optimizar la estructura de financiamiento y reducir presión de pasivos.",
                        "Liquidez": "Prioridad: mejorar capital de trabajo (cobranza, inventario y gestión de pagos).",
                    }

                    def trend_label(value: float | None) -> str:
                        if value is None:
                            return "sin tendencia clara"
                        if value > 1.0:
                            return "tendencia favorable"
                        if value < -1.0:
                            return "tendencia desfavorable"
                        return "tendencia estable"

                    best_block_trend_text = trend_label(block_trend_scores.get(best_block))
                    worst_block_trend_text = trend_label(block_trend_scores.get(worst_block))
                    narrative_1 = (
                        f"En <strong>{selected_company}</strong>, el análisis automático ubica el desempeño en nivel "
                        f"<strong>{score_label(total_score)}</strong>, con fortaleza relativa en "
                        f"<strong>{block_names[best_block]}</strong> ({best_metric['name']}: {best_metric['value_text']}) "
                        f"y <strong>{best_block_trend_text}</strong> en ese frente."
                    )
                    narrative_2 = (
                        f"La principal señal de riesgo se concentra en <strong>{block_names[worst_block]}</strong>, "
                        f"especialmente en {worst_metric['name']} ({worst_metric['value_text']}), con "
                        f"<strong>{worst_block_trend_text}</strong>."
                    )
                    narrative_3 = priority_map[worst_block].replace("Prioridad: ", "En el corto plazo, se recomienda ")
                    pdf_total_score = total_score
                    pdf_status_label = score_label(total_score)
                    pdf_narratives = [narrative_1, narrative_2, narrative_3]

                    if total_score >= 75:
                        status_bg, status_border, status_text = "#dcfce7", "#86efac", "#14532d"
                    elif total_score >= 60:
                        status_bg, status_border, status_text = "#dbeafe", "#93c5fd", "#1e3a8a"
                    elif total_score >= 45:
                        status_bg, status_border, status_text = "#fef3c7", "#fcd34d", "#78350f"
                    else:
                        status_bg, status_border, status_text = "#fee2e2", "#fca5a5", "#7f1d1d"

                    st.markdown(
                        f"""
                        <div style="border:1px solid #c7d2fe; border-radius:14px; padding:14px 16px; margin:0.2rem 0 0.8rem 0; background:linear-gradient(180deg,#f8faff 0%,#ffffff 100%);">
                            <div style="display:flex; justify-content:space-between; align-items:center; gap:0.8rem; flex-wrap:wrap;">
                                <div style="font-size:1.05rem; font-weight:700; color:#111827;">Resumen Ejecutivo Automático</div>
                                <div style="display:flex; align-items:center; gap:0.5rem;">
                                    <span style="padding:0.28rem 0.6rem; border-radius:999px; background:#eef2ff; color:#3730a3; font-size:0.85rem; font-weight:600;">
                                        Score total: {total_score:.1f}/100
                                    </span>
                                    <span style="padding:0.28rem 0.6rem; border-radius:999px; background:{status_bg}; border:1px solid {status_border}; color:{status_text}; font-size:0.85rem; font-weight:700;">
                                        {score_label(total_score).capitalize()}
                                    </span>
                                </div>
                            </div>
                            <div style="margin-top:0.75rem; color:#1f2937; line-height:1.55; font-size:0.95rem;">
                                <div>{narrative_1}</div>
                                <div style="margin-top:0.3rem;">{narrative_2}</div>
                                <div style="margin-top:0.3rem;">{narrative_3}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    score_cols = st.columns(3)
                    for idx, block in enumerate(["Rentabilidad", "Estructura", "Liquidez"]):
                        block_score = block_scores.get(block)
                        block_score_text = f"{block_score:.1f}/100" if block_score is not None else "N/D"
                        if block_score is None:
                            block_style = ("#f3f4f6", "#d1d5db", "#374151")
                        elif block_score >= 75:
                            block_style = ("#dcfce7", "#86efac", "#14532d")
                        elif block_score >= 60:
                            block_style = ("#dbeafe", "#93c5fd", "#1e3a8a")
                        elif block_score >= 45:
                            block_style = ("#fef3c7", "#fcd34d", "#78350f")
                        else:
                            block_style = ("#fee2e2", "#fca5a5", "#7f1d1d")

                        with score_cols[idx]:
                            st.markdown(
                                f"""
                                <div style="border:1px solid {block_style[1]}; background:{block_style[0]}; border-radius:10px; padding:9px 10px;">
                                    <div style="font-size:0.82rem; color:#374151; font-weight:600;">{block}</div>
                                    <div style="font-size:1.05rem; color:{block_style[2]}; font-weight:700; margin-top:0.1rem;">{block_score_text}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                else:
                    st.info("No hay datos suficientes para construir el resumen automático de indicadores.")

                indicator_rows = []
                for indicator in indicator_list:
                    indicator_rows.append(
                        {
                            "Indicador": indicator,
                            "2021": annual_indicators_df.loc[2021, indicator] if indicator in annual_indicators_df.columns else pd.NA,
                            "2022": annual_indicators_df.loc[2022, indicator] if indicator in annual_indicators_df.columns else pd.NA,
                            "2023": annual_indicators_df.loc[2023, indicator] if indicator in annual_indicators_df.columns else pd.NA,
                            "2024": annual_indicators_df.loc[2024, indicator] if indicator in annual_indicators_df.columns else pd.NA,
                        }
                    )

                indicators_table_df = pd.DataFrame(indicator_rows)
                percent_indicators = {
                    "MARGEN BRUTO",
                    "MARGEN EBITDA",
                    "MARGEN DE UTILIDAD",
                    "ROA",
                    "ROE",
                    "ENDEUDAMIENTO",
                }

                indicators_display_df = indicators_table_df.copy()
                for year_col in ["2021", "2022", "2023", "2024"]:
                    indicators_display_df[year_col] = indicators_display_df.apply(
                        lambda row: (
                            f"{row[year_col] * 100:.2f}%"
                            if row["Indicador"] in percent_indicators and pd.notna(row[year_col])
                            else (f"{row[year_col]:,.2f}" if pd.notna(row[year_col]) else "-")
                        ),
                        axis=1,
                    )

                indicator_groups = {
                    "INDICADORES DE RENTABILIDAD": [
                        "MARGEN BRUTO",
                        "MARGEN EBITDA",
                        "MARGEN DE UTILIDAD",
                        "ROE",
                        "ROA",
                    ],
                    "INDICADORES DE ENDEUDAMIENTO": [
                        "ENDEUDAMIENTO",
                    ],
                    "INDICADORES DE LIQUIDEZ": [
                        "DÍAS DE INVENTARIO",
                        "DÍAS DE COBRO",
                        "DÍAS DE PAGO",
                        "CICLO DE CONVERSIÓN DE EFECTIVO",
                        "RAZÓN CORRIENTE",
                        "PRUEBA ÁCIDA",
                    ],
                }

                display_map = indicators_display_df.set_index("Indicador").to_dict("index")
                grouped_rows = []
                for group_label, group_indicators in indicator_groups.items():
                    grouped_rows.append(
                        {
                            "_is_group": True,
                            "Indicador": group_label,
                            "2021": "",
                            "2022": "",
                            "2023": "",
                            "2024": "",
                        }
                    )
                    for indicator in group_indicators:
                        row_values = display_map.get(
                            indicator,
                            {"2021": "-", "2022": "-", "2023": "-", "2024": "-"},
                        )
                        grouped_rows.append(
                            {
                                "_is_group": False,
                                "Indicador": indicator,
                                "2021": row_values["2021"],
                                "2022": row_values["2022"],
                                "2023": row_values["2023"],
                                "2024": row_values["2024"],
                            }
                        )

                indicators_grouped_df = pd.DataFrame(grouped_rows)
                group_rows = set(indicators_grouped_df.index[indicators_grouped_df["_is_group"]].tolist())
                indicators_view_df = indicators_grouped_df.drop(columns=["_is_group"])

                def style_indicator_rows(row: pd.Series) -> list[str]:
                    if row.name in group_rows:
                        return [
                            "font-weight: 700; background-color: #eef2ff; border-top: 2px solid #c7d2fe; color: #1f2937;"
                        ] * len(row)
                    return ["background-color: #ffffff;"] * len(row)

                def style_indicator_label(row: pd.Series) -> list[str]:
                    if row.name in group_rows:
                        return ["letter-spacing: 0.02em;"]
                    return ["padding-left: 1.2rem; color: #374151;"]

                styled_indicators = (
                    indicators_view_df.style
                    .apply(style_indicator_rows, axis=1)
                    .apply(style_indicator_label, axis=1, subset=["Indicador"])
                    .set_properties(subset=["2021", "2022", "2023", "2024"], **{"text-align": "right"})
                )
                st.dataframe(styled_indicators, width="stretch", hide_index=True)
                try:
                    indicators_pdf_bytes = to_indicators_pdf_bytes(
                        company_name=selected_company,
                        ruc=str(ruc),
                        total_score=pdf_total_score,
                        status_label=pdf_status_label,
                        narrative_lines=pdf_narratives,
                        block_scores=pdf_block_scores,
                        indicators_table_df=indicators_view_df,
                    )
                    indicators_pdf_file = f"indicadores_financieros_clave_{safe_filename(selected_company)}.pdf"
                    st.download_button(
                        "Descargar PDF - Indicadores Financieros Clave",
                        data=indicators_pdf_bytes,
                        file_name=indicators_pdf_file,
                        mime="application/pdf",
                        key=f"download_pdf_ind_{safe_filename(selected_company)}",
                    )
                except Exception:
                    st.error("No se pudo generar el PDF de Indicadores.")

                if missing_indicators:
                    st.warning(f"Indicadores no encontrados en el dataset: {', '.join(missing_indicators)}")

    with tab_graph:
        st.markdown("#### Gráficos Seleccionados")

        st.markdown("**Gráfico 1: Evolución de Ingresos, Costo de Ventas y Utilidad Bruta**")
        utilidad_bruta_series = None
        if "UTILIDAD BRUTA" in annual_df.columns:
            utilidad_bruta_series = annual_df["UTILIDAD BRUTA"]
        elif "CONTRIBUCIÓN MARGINAL" in annual_df.columns:
            utilidad_bruta_series = annual_df["CONTRIBUCIÓN MARGINAL"]
        elif "INGRESOS" in annual_df.columns and "COSTO DE VENTAS" in annual_df.columns:
            utilidad_bruta_series = annual_df["INGRESOS"] - annual_df["COSTO DE VENTAS"]

        graph_1_df = pd.DataFrame(
            {
                "INGRESOS": annual_df["INGRESOS"] if "INGRESOS" in annual_df.columns else pd.NA,
                "COSTO DE VENTAS": annual_df["COSTO DE VENTAS"] if "COSTO DE VENTAS" in annual_df.columns else pd.NA,
                "UTILIDAD BRUTA": utilidad_bruta_series if utilidad_bruta_series is not None else pd.NA,
            },
            index=annual_df.index,
        )
        graph_1_df.index.name = "AÑO"
        if graph_1_df.dropna(how="all").empty:
            st.warning("No hay datos suficientes para el Grafico 1.")
        else:
            graph_1_order = ["INGRESOS", "COSTO DE VENTAS", "UTILIDAD BRUTA"]
            graph_1_long = (
                graph_1_df.reset_index()
                .melt(
                    id_vars="AÑO",
                    value_vars=graph_1_order,
                    var_name="Cuenta",
                    value_name="Valor",
                )
                .dropna(subset=["Valor"])
            )
            chart_1 = (
                alt.Chart(graph_1_long)
                .mark_bar()
                .encode(
                    x=alt.X("AÑO:O", title="AÑO"),
                    xOffset=alt.XOffset("Cuenta:N", sort=graph_1_order),
                    y=alt.Y("Valor:Q", title="Valor"),
                    color=alt.Color(
                        "Cuenta:N",
                        title="Cuenta",
                        scale=alt.Scale(domain=graph_1_order),
                    ),
                    tooltip=["AÑO:O", "Cuenta:N", alt.Tooltip("Valor:Q", format=",.0f")],
                )
                .properties(height=320)
            )
            st.altair_chart(chart_1, use_container_width=True)

        st.markdown("**Gráfico 2: Evolución del Balance**")
        balance_company_df_graph = balance_data[balance_data["RUC"] == str(ruc)].copy()
        if balance_company_df_graph.empty:
            balance_company_df_graph = balance_data[balance_data["NOMBRE"] == selected_company].copy()

        graph_2_required_cols = ["ACTIVO", "PASIVO", "PATRIMONIO"]
        missing_graph_2_cols = [col for col in graph_2_required_cols if col not in balance_company_df_graph.columns]
        if balance_company_df_graph.empty or missing_graph_2_cols:
            st.warning("No hay datos suficientes para el Grafico 2.")
        else:
            annual_balance_graph_df = (
                balance_company_df_graph.groupby("AÑO", dropna=False)[graph_2_required_cols]
                .sum(numeric_only=True)
                .sort_index()
                .reset_index()
            )

            graph_2_bars_long = annual_balance_graph_df.melt(
                id_vars="AÑO",
                value_vars=["PASIVO", "PATRIMONIO"],
                var_name="Cuenta",
                value_name="Valor",
            )
            graph_2_bars_long = graph_2_bars_long.dropna(subset=["Valor"])
            graph_2_activo = annual_balance_graph_df[["AÑO", "ACTIVO", "PASIVO", "PATRIMONIO"]].dropna(subset=["ACTIVO"])

            if graph_2_bars_long.empty or graph_2_activo.empty:
                st.warning("No hay datos suficientes para el Grafico 2.")
            else:
                st.markdown(
                    """
                    <div style="display:flex; gap:1.2rem; align-items:center; margin: 0.2rem 0 0.6rem 0;">
                        <span style="display:inline-flex; align-items:center; gap:0.45rem;">
                            <span style="width:14px; height:14px; border:2px solid #111827; background:transparent; display:inline-block;"></span>
                            <span style="font-size:0.92rem; color:#111827;">ACTIVO</span>
                        </span>
                        <span style="display:inline-flex; align-items:center; gap:0.45rem;">
                            <span style="width:14px; height:14px; background:#60a5fa; display:inline-block;"></span>
                            <span style="font-size:0.92rem; color:#111827;">PASIVO</span>
                        </span>
                        <span style="display:inline-flex; align-items:center; gap:0.45rem;">
                            <span style="width:14px; height:14px; background:#34d399; display:inline-block;"></span>
                            <span style="font-size:0.92rem; color:#111827;">PATRIMONIO</span>
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                graph_2_color = alt.Color(
                    "Cuenta:N",
                    title="Cuenta",
                    scale=alt.Scale(
                        domain=["PASIVO", "PATRIMONIO"],
                        range=["#60a5fa", "#34d399"],
                    ),
                    legend=None,
                )
                chart_2_bars = (
                    alt.Chart(graph_2_bars_long)
                    .mark_bar()
                    .encode(
                        x=alt.X("AÑO:O", title="AÑO"),
                        y=alt.Y("Valor:Q", title="Valor"),
                        color=graph_2_color,
                        tooltip=["AÑO:O", "Cuenta:N", alt.Tooltip("Valor:Q", format=",.0f")],
                    )
                )
                chart_2_activo = (
                    alt.Chart(graph_2_activo)
                    .mark_bar(fillOpacity=0, stroke="#111827", strokeWidth=2)
                    .encode(
                        x=alt.X("AÑO:O", title="AÑO"),
                        y=alt.Y("ACTIVO:Q", title="Valor"),
                        color=alt.value("#111827"),
                        tooltip=[
                            "AÑO:O",
                            alt.Tooltip("ACTIVO:Q", title="ACTIVO", format=",.0f"),
                            alt.Tooltip("PASIVO:Q", title="PASIVO", format=",.0f"),
                            alt.Tooltip("PATRIMONIO:Q", title="PATRIMONIO", format=",.0f"),
                        ],
                    )
                )
                chart_2 = (chart_2_bars + chart_2_activo).properties(height=320)
                st.altair_chart(chart_2, use_container_width=True)

        st.markdown("**Gráfico 3: Evolución de ROE y ROA**")
        if indicators_data is None:
            st.warning("No hay datos de indicadores para el Grafico 3.")
        else:
            indicators_company_df_graph = pd.DataFrame()
            if "RUC" in indicators_data.columns:
                indicators_company_df_graph = indicators_data[indicators_data["RUC"] == str(ruc)].copy()
            if indicators_company_df_graph.empty and "NOMBRE" in indicators_data.columns:
                indicators_company_df_graph = indicators_data[indicators_data["NOMBRE"] == selected_company].copy()

            graph_3_columns = [col for col in ["ROE", "ROA"] if col in indicators_company_df_graph.columns]
            if indicators_company_df_graph.empty or "AÑO" not in indicators_company_df_graph.columns or not graph_3_columns:
                st.warning("No hay datos suficientes para el Grafico 3.")
            else:
                annual_indicators_graph_df = (
                    indicators_company_df_graph.groupby("AÑO", dropna=False)[graph_3_columns]
                    .mean(numeric_only=True)
                    .sort_index()
                    .reset_index()
                )
                for col in graph_3_columns:
                    annual_indicators_graph_df[col] = annual_indicators_graph_df[col] * 100
                graph_3_long = annual_indicators_graph_df.melt(
                    id_vars="AÑO",
                    value_vars=graph_3_columns,
                    var_name="Indicador",
                    value_name="Porcentaje",
                )
                graph_3_long = graph_3_long.dropna(subset=["Porcentaje"])

                if graph_3_long.empty:
                    st.warning("No hay datos suficientes para el Grafico 3.")
                else:
                    chart_3 = (
                        alt.Chart(graph_3_long)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("AÑO:O", title="AÑO"),
                            y=alt.Y("Porcentaje:Q", title="%"),
                            color=alt.Color("Indicador:N", title="Indicador"),
                            tooltip=[
                                "AÑO:O",
                                "Indicador:N",
                                alt.Tooltip("Porcentaje:Q", format=".2f"),
                            ],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(chart_3, use_container_width=True)

        st.markdown("**Gráfico 4: Evolución de Indicadores de Gestión de Capital de Trabajo**")
        if indicators_data is None:
            st.warning("No hay datos de indicadores para el Grafico 4.")
        else:
            indicators_company_df_graph = pd.DataFrame()
            if "RUC" in indicators_data.columns:
                indicators_company_df_graph = indicators_data[indicators_data["RUC"] == str(ruc)].copy()
            if indicators_company_df_graph.empty and "NOMBRE" in indicators_data.columns:
                indicators_company_df_graph = indicators_data[indicators_data["NOMBRE"] == selected_company].copy()

            graph_4_columns = [
                "DÍAS DE INVENTARIO",
                "DÍAS DE COBRO",
                "DÍAS DE PAGO",
                "CICLO DE CONVERSIÓN DE EFECTIVO",
            ]
            available_graph_4_cols = [col for col in graph_4_columns if col in indicators_company_df_graph.columns]
            if indicators_company_df_graph.empty or "AÑO" not in indicators_company_df_graph.columns or not available_graph_4_cols:
                st.warning("No hay datos suficientes para el Grafico 4.")
            else:
                annual_graph_4_df = (
                    indicators_company_df_graph.groupby("AÑO", dropna=False)[available_graph_4_cols]
                    .mean(numeric_only=True)
                    .sort_index()
                    .reset_index()
                )
                graph_4_long = annual_graph_4_df.melt(
                    id_vars="AÑO",
                    value_vars=available_graph_4_cols,
                    var_name="Indicador",
                    value_name="Valor",
                )
                graph_4_long = graph_4_long.dropna(subset=["Valor"])

                if graph_4_long.empty:
                    st.warning("No hay datos suficientes para el Grafico 4.")
                else:
                    chart_4 = (
                        alt.Chart(graph_4_long)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("AÑO:O", title="AÑO"),
                            y=alt.Y("Valor:Q", title="Dias"),
                            color=alt.Color("Indicador:N", title="Indicador"),
                            tooltip=[
                                "AÑO:O",
                                "Indicador:N",
                                alt.Tooltip("Valor:Q", format=".2f"),
                            ],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(chart_4, use_container_width=True)
else:
    st.info("Selecciona una empresa para visualizar su analisis financiero.")
