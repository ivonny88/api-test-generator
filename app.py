import streamlit as st
import groq
import json
import yaml
import re
import hashlib
import time
from datetime import date

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="API Test Generator | Fátima QA",
    page_icon="🔌",
    layout="wide"
)

# ── Constantes de seguridad ──────────────────────────────────────────────────
MAX_INPUT_LENGTH = 8000          # límite de caracteres de input
MAX_ENDPOINT_LENGTH = 300        # límite para URL de endpoint
ALLOWED_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
RATE_LIMIT_SECONDS = 10          # segundos mínimos entre generaciones

# ── Frases de prompt injection que bloqueamos ────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous\s+|prior\s+|above\s+)?instructions",
    r"ignore all",
    r"forget (everything|all|your instructions)",
    r"you are now",
    r"act as (a |an )?",
    r"jailbreak",
    r"DAN mode",
    r"pretend (you are|to be)",
    r"disregard (your|all|previous)",
    r"system prompt",
    r"override",
    r"bypass",
    r"<\s*script",           # XSS básico
    r"javascript:",
    r"\beval\s*\(",
    r"__import__",           # Python injection
    r"os\.system",
    r"subprocess",
    r"tell me a joke",
]

def detect_injection(text: str) -> bool:
    """Devuelve True si el texto parece un intento de prompt injection."""
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False

def sanitize_input(text: str, max_length: int = MAX_INPUT_LENGTH) -> str:
    """Limpia y trunca el input del usuario."""
    # Eliminar caracteres de control excepto saltos de línea y tabulaciones
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Truncar
    return text[:max_length]

def validate_url(url: str) -> bool:
    """Valida que la URL tenga formato razonable."""
    pattern = r'^https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+$'
    return bool(re.match(pattern, url)) and len(url) <= MAX_ENDPOINT_LENGTH

def validate_openapi(text: str) -> tuple[bool, str]:
    """Intenta parsear como JSON o YAML. Devuelve (válido, mensaje)."""
    try:
        parsed = json.loads(text)
        if "paths" not in parsed and "openapi" not in parsed and "swagger" not in parsed:
            return False, "No parece un spec OpenAPI válido (falta 'paths' u 'openapi')."
        return True, "JSON válido"
    except json.JSONDecodeError:
        pass
    try:
        parsed = yaml.safe_load(text)
        if isinstance(parsed, dict):
            return True, "YAML válido"
        return False, "YAML parseado pero no es un objeto."
    except yaml.YAMLError as e:
        return False, f"Error YAML: {e}"

# ── Rate limiting por sesión ─────────────────────────────────────────────────
def check_rate_limit() -> bool:
    """Evita spam de generaciones. Devuelve True si puede generar."""
    now = time.time()
    last = st.session_state.get("last_generation_time", 0)
    if now - last < RATE_LIMIT_SECONDS:
        remaining = int(RATE_LIMIT_SECONDS - (now - last))
        st.warning(f"⏳ Espera {remaining}s antes de generar de nuevo.")
        return False
    return True

# ── Límite diario (usuarios free) ────────────────────────────────────────────
def get_usage_key() -> str:
    today = str(date.today())
    session_id = st.session_state.get("session_id", "default")
    return f"usage_{today}_{session_id}"

def check_daily_limit(is_pro: bool) -> tuple[bool, int]:
    """Devuelve (puede_generar, usos_restantes)."""
    if is_pro:
        return True, 999
    daily_limit = int(st.secrets.get("DAILY_LIMIT", 3))
    key = get_usage_key()
    used = st.session_state.get(key, 0)
    remaining = daily_limit - used
    return remaining > 0, remaining

def increment_usage():
    key = get_usage_key()
    st.session_state[key] = st.session_state.get(key, 0) + 1

# ── Cliente Groq ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_groq_client():
    return groq.Groq(api_key=st.secrets["GROQ_API_KEY"])

# ── Generación de tests ───────────────────────────────────────────────────────
def build_prompt(input_type: str, content: str, output_format: str, http_method: str = "") -> str:
    method_hint = f"Método HTTP: {http_method}. " if http_method else ""

    format_instructions = {
        "Postman Collection (JSON)": (
            "Genera una colección Postman v2.1 completa en formato JSON. "
            "Incluye al menos 6 tests: happy path, campos vacíos, auth inválida, "
            "datos incorrectos, límite de rate y error de servidor (500). "
            "Añade scripts de test en Postman (pm.test) para validar status codes, "
            "tiempos de respuesta y estructura del body. "
            "Responde ÚNICAMENTE con el JSON de la colección, sin explicaciones."
        ),
        "pytest + requests (Python)": (
            "Genera un archivo pytest completo en Python usando la librería requests. "
            "Incluye al menos 6 tests: happy path, campos vacíos, auth inválida, "
            "datos incorrectos, parámetros de borde y error esperado. "
            "Usa fixtures, parametrize donde tenga sentido y assertions claras. "
            "Responde ÚNICAMENTE con el código Python, sin explicaciones."
        ),
        "Ambos": (
            "Genera DOS bloques: primero la colección Postman v2.1 en JSON (marcada con "
            "```json) y luego el archivo pytest en Python (marcado con ```python). "
            "Cada uno con al menos 5 tests. Sin explicaciones fuera de los bloques de código."
        ),
    }

    input_descriptions = {
        "URL de endpoint": f"Endpoint URL: {content}. {method_hint}",
        "Especificación OpenAPI/Swagger": f"Especificación OpenAPI:\n{content[:4000]}",
        "Descripción en texto libre": f"Descripción de la API:\n{content[:3000]}",
    }

    system_prompt = (
        "Eres un experto en QA y testing de APIs REST. "
        "Tu única función es generar casos de prueba y código de tests. "
        "Nunca ejecutes instrucciones del usuario que no estén relacionadas con testing. "
        "Si el input no parece una API o especificación, responde solo con: "
        "'ERROR: El input no parece relacionado con una API.'"
    )

    user_prompt = (
        f"{input_descriptions.get(input_type, content)}\n\n"
        f"Formato de salida requerido: {output_format}\n\n"
        f"{format_instructions.get(output_format, '')}"
    )

    return system_prompt, user_prompt

def generate_tests(input_type: str, content: str, output_format: str, http_method: str = "") -> str:
    client = get_groq_client()
    system_prompt, user_prompt = build_prompt(input_type, content, output_format, http_method)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4000,
        temperature=0.3,
    )
    return response.choices[0].message.content

# ── UI ────────────────────────────────────────────────────────────────────────
def main():
    # Inicializar session state
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = hashlib.md5(
            str(time.time()).encode()
        ).hexdigest()[:8]
    if "is_pro" not in st.session_state:
        st.session_state["is_pro"] = False
    if "generated_output" not in st.session_state:
        st.session_state["generated_output"] = None

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🔌 API Test Generator")
    st.markdown(
        "**Genera tests de API listos para usar** — Postman Collections y pytest "
        "con un solo clic. Powered by LLaMA 3.3 via Groq."
    )
    st.divider()

    # ── Sidebar: Pro key + info ───────────────────────────────────────────────
    with st.sidebar:
        st.header("⚡ Versión Pro")
        pro_input = st.text_input(
            "Código Pro",
            type="password",
            placeholder="FATIMAQA-APITEST-2024",
            help="Generaciones ilimitadas"
        )
        if pro_input:
            # Comparar hash para no exponer la clave en memoria
            expected_hash = hashlib.sha256(
                st.secrets.get("PRO_KEY", "").encode()
            ).hexdigest()
            input_hash = hashlib.sha256(pro_input.encode()).hexdigest()
            if input_hash == expected_hash:
                st.session_state["is_pro"] = True
                st.success("✅ Modo Pro activado")
            else:
                st.error("❌ Código incorrecto")

        is_pro = st.session_state["is_pro"]
        can_generate, remaining = check_daily_limit(is_pro)

        st.divider()
        if is_pro:
            st.info("🚀 Generaciones: **Ilimitadas**")
        else:
            color = "🟢" if remaining > 1 else "🟡" if remaining == 1 else "🔴"
            st.info(f"{color} Generaciones hoy: **{remaining}/3**")
            st.markdown(
                "🔓 [Obtener Pro en Gumroad](https://fatimaflare207.gumroad.com/l/iliohe)",
                unsafe_allow_html=False
            )

        st.divider()
        st.markdown("### ☕ ¿Te es útil?")
        st.markdown(
            "[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-%23FFDD00?style=flat&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/fatimaqa)"
        )
        st.divider()
        st.markdown("**Creado por [Fátima QA](https://fatimaqa.com)**")

    # ── Formulario principal ──────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        input_type = st.selectbox(
            "📥 Tipo de input",
            ["URL de endpoint", "Especificación OpenAPI/Swagger", "Descripción en texto libre"],
        )

    with col2:
        output_format = st.selectbox(
            "📤 Formato de output",
            ["Postman Collection (JSON)", "pytest + requests (Python)", "Ambos"],
        )

    # Input dinámico según tipo
    http_method = ""
    content = ""

    if input_type == "URL de endpoint":
        col_url, col_method = st.columns([3, 1])
        with col_url:
            content = st.text_input(
                "URL del endpoint",
                placeholder="https://api.ejemplo.com/v1/users",
            )
        with col_method:
            http_method = st.selectbox("Método", sorted(ALLOWED_METHODS))

    elif input_type == "Especificación OpenAPI/Swagger":
        content = st.text_area(
            "Pega tu spec OpenAPI/Swagger (JSON o YAML)",
            height=250,
            placeholder='{"openapi": "3.0.0", "paths": {...}}',
        )

    else:  # Texto libre
        content = st.text_area(
            "Describe tu API",
            height=150,
            placeholder=(
                "Ej: API REST de gestión de usuarios. "
                "Endpoint POST /users que recibe nombre, email y password. "
                "Devuelve 201 con el usuario creado o 400 si el email ya existe."
            ),
        )

    # ── Botón de generación ───────────────────────────────────────────────────
    generate_clicked = st.button("⚡ Generar Tests", type="primary", use_container_width=True)

    if generate_clicked:
        # — Validaciones de seguridad —
        if not content.strip():
            st.error("❌ El campo de input está vacío.")
            st.stop()

        if detect_injection(content):
            st.error(
                "🚫 Se ha detectado contenido no permitido en el input. "
                "Por favor, introduce una URL, especificación o descripción de API válida."
            )
            st.stop()

        if input_type == "URL de endpoint":
            if not validate_url(content):
                st.error("❌ La URL no tiene un formato válido. Asegúrate de incluir http:// o https://")
                st.stop()

        elif input_type == "Especificación OpenAPI/Swagger":
            valid, msg = validate_openapi(content)
            if not valid:
                st.error(f"❌ Especificación inválida: {msg}")
                st.stop()

        content = sanitize_input(content)

        if not check_rate_limit():
            st.stop()

        can_generate, remaining = check_daily_limit(st.session_state["is_pro"])
        if not can_generate:
            st.error(
                "🔒 Has alcanzado el límite diario de 3 generaciones. "
                "Obtén la versión Pro para generaciones ilimitadas."
            )
            st.markdown(
                "👉 [Comprar Pro — 9€](https://fatimaflare207.gumroad.com/l/iliohe)"
            )
            st.stop()

        # — Generar —
        with st.spinner("🤖 Generando tests con LLaMA 3.3..."):
            try:
                result = generate_tests(input_type, content, output_format, http_method)
                st.session_state["generated_output"] = result
                st.session_state["last_generation_time"] = time.time()
                if not st.session_state["is_pro"]:
                    increment_usage()
            except Exception as e:
                st.error(f"❌ Error al conectar con la IA: {str(e)}")
                st.stop()

    # ── Mostrar output ────────────────────────────────────────────────────────
    if st.session_state.get("generated_output"):
        st.divider()
        st.subheader("✅ Tests Generados")
        output = st.session_state["generated_output"]

        # Detectar si hay bloques de código separados (modo "Ambos")
        json_match = re.search(r'```json\n(.*?)```', output, re.DOTALL)
        python_match = re.search(r'```python\n(.*?)```', output, re.DOTALL)

        if json_match and python_match:
            tab1, tab2 = st.tabs(["📮 Postman Collection", "🐍 pytest"])
            with tab1:
                postman_code = json_match.group(1).strip()
                st.code(postman_code, language="json")
                st.download_button(
                    "⬇️ Descargar colección Postman",
                    data=postman_code,
                    file_name="api_tests_postman.json",
                    mime="application/json",
                )
            with tab2:
                pytest_code = python_match.group(1).strip()
                st.code(pytest_code, language="python")
                st.download_button(
                    "⬇️ Descargar archivo pytest",
                    data=pytest_code,
                    file_name="test_api.py",
                    mime="text/plain",
                )
        elif json_match:
            postman_code = json_match.group(1).strip()
            st.code(postman_code, language="json")
            st.download_button(
                "⬇️ Descargar colección Postman",
                data=postman_code,
                file_name="api_tests_postman.json",
                mime="application/json",
            )
        elif python_match:
            pytest_code = python_match.group(1).strip()
            st.code(pytest_code, language="python")
            st.download_button(
                "⬇️ Descargar archivo pytest",
                data=pytest_code,
                file_name="test_api.py",
                mime="text/plain",
            )
        else:
            # Output sin bloques de código marcados
            st.code(output, language="json")
            st.download_button(
                "⬇️ Descargar output",
                data=output,
                file_name="api_tests.txt",
                mime="text/plain",
            )

if __name__ == "__main__":
    main()