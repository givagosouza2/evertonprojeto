import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math

st.set_page_config(layout="wide")
st.title("Análise Vetorial + Elipse 95% (coordenadas espaciais)")

st.markdown("""
Este aplicativo calcula vetores a partir de coordenadas X e Y em série temporal e extrai:
- **Número de toques**
- **Intervalo médio entre toques**
- **Somatória da resultante espacial** (∑‖Δr‖)
- **Área da elipse que cobre 95%** das coordenadas espaciais (X,Y)
Além da análise vetorial (ΔX, ΔY) com elipse de inércia e S-index.
""")

uploaded_file = st.file_uploader(
    "📄 Carregue um arquivo .txt ou .csv com colunas: Tempo, X, Y, ...", type=["txt", "csv"]
)

def _to_numeric(series):
    """Converte para numérico e remove NaN."""
    return pd.to_numeric(series, errors="coerce")

def calcular_elipse_inercia(x, y):
    """
    Elipse baseada na covariância (PCA).
    Retorna eixos (1-sigma), razão, ângulo, s-index, autovetores e autovalores.
    """
    X = np.vstack([x, y])
    cov = np.cov(X)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # 1-sigma (desvio padrão) em cada eixo principal
    eixo_maior = np.sqrt(max(eigvals[0], 0))
    eixo_menor = np.sqrt(max(eigvals[1], 0))

    razao = eixo_maior / eixo_menor if eixo_menor != 0 else np.inf
    angulo = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    s_index = razao
    return eixo_maior, eixo_menor, razao, angulo, s_index, eigvecs, eigvals

def elipse_95_xy(x, y):
    """
    Elipse de confiança 95% para dados 2D assumindo normalidade:
    (p - mu)^T Sigma^{-1} (p - mu) <= chi2(0.95, df=2)
    Para df=2, chi2_0.95 ≈ 5.991.
    """
    chi2_95_df2 = 5.991464547107979  # constante (95%, df=2)

    mu_x, mu_y = np.mean(x), np.mean(y)
    _, _, _, ang, _, eigvecs, eigvals = calcular_elipse_inercia(x, y)

    # Semi-eixos (raios) da elipse 95%
    # sqrt(chi2) * sqrt(lambda)  (lambda = autovalor)
    a = np.sqrt(chi2_95_df2 * max(eigvals[0], 0))  # semi-eixo maior
    b = np.sqrt(chi2_95_df2 * max(eigvals[1], 0))  # semi-eixo menor

    # Área = pi*a*b
    area = math.pi * a * b
    return (mu_x, mu_y), a, b, ang, area, eigvals, eigvecs

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=None, engine="python")

    if df.shape[1] < 3:
        st.error("O arquivo deve conter pelo menos três colunas: tempo, X e Y.")
        st.stop()

    # Tempo, X, Y (por posição no seu arquivo)
    t_raw = _to_numeric(df.iloc[:, 0])
    x_raw = _to_numeric(df.iloc[:, 1])
    y_raw = _to_numeric(df.iloc[:, 2])

    valid = (~t_raw.isna()) & (~x_raw.isna()) & (~y_raw.isna())
    dfv = pd.DataFrame({"t": t_raw[valid], "x": x_raw[valid], "y": y_raw[valid]}).reset_index(drop=True)

    if len(dfv) < 3:
        st.error("Poucos pontos válidos após limpeza. Verifique o arquivo.")
        st.stop()

    t = dfv["t"].to_numpy()
    x = dfv["x"].to_numpy()
    y = dfv["y"].to_numpy()

    # --- Métricas solicitadas ---
    n_toques = len(x)  # número de pontos (toques)

    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]  # ignora zeros/negativos (caso tempo esteja duplicado ou invertido)
    intervalo_medio = float(np.mean(dt)) if len(dt) else np.nan

    dx = np.diff(x)
    dy = np.diff(y)

    # resultante espacial por passo e somatória
    dr = np.sqrt(dx**2 + dy**2)
    soma_resultante_espacial = float(np.nansum(dr))

    # --- Elipse 95% das coordenadas espaciais (X,Y) ---
    (cx, cy), a95, b95, ang95, area95, eigvals_xy, eigvecs_xy = elipse_95_xy(x, y)

    # --- Elipse (ΔX, ΔY) do seu pipeline original ---
    eixo_maior, eixo_menor, razao, angulo, s_index, eigvecs, eigvals = calcular_elipse_inercia(dx, dy)

    st.subheader("📊 Métricas principais")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nº de toques", f"{n_toques}")
    c2.metric("Intervalo médio entre toques", f"{intervalo_medio:.4f}" if np.isfinite(intervalo_medio) else "—")
    c3.metric("∑ resultante espacial (∑‖Δr‖)", f"{soma_resultante_espacial:.4f}")
    c4.metric("Área elipse 95% (X,Y)", f"{area95:.4f}")

    st.subheader("📌 Elipse 95% nas coordenadas espaciais (X,Y)")
    st.write(f"Centro (média): ({cx:.2f}, {cy:.2f})")
    st.write(f"Semi-eixo maior (95%): {a95:.4f}")
    st.write(f"Semi-eixo menor (95%): {b95:.4f}")
    st.write(f"Ângulo (graus): {ang95:.2f}°")

    st.subheader("📈 Elipse vetorial (ΔX, ΔY) e S-index")
    st.write(f"**Eixo maior (1σ):** {eixo_maior:.4f}")
    st.write(f"**Eixo menor (1σ):** {eixo_menor:.4f}")
    st.write(f"**Razão entre eixos:** {razao:.4f}")
    st.write(f"**Ângulo (graus):** {angulo:.2f}°")
    st.write(f"**S-index:** {s_index:.4f}")

    # --- Plots ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Passo 1: Posições (X,Y) + Elipse 95%")
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.plot(x, y, "o-", alpha=0.6, label="Trajetória")

        # Elipse 95% em X,Y
        ellipse_xy = Ellipse(
            (cx, cy),
            width=2 * a95,   # largura = 2*semi-eixo
            height=2 * b95,  # altura = 2*semi-eixo
            angle=ang95,
            edgecolor="red",
            fc="None",
            lw=2,
            label="Elipse 95% (X,Y)"
        )
        ax1.add_patch(ellipse_xy)

        ax1.set_title("Coordenadas espaciais (X, Y) + Elipse 95%")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_aspect("equal", adjustable="datalim")
        ax1.legend()
        st.pyplot(fig1)

    with col2:
        st.subheader("📈 Passo 2: Vetores (ΔX,ΔY) + Elipse de inércia")
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.scatter(dx, dy, alpha=0.6, label="Vetores de deslocamento")

        ax2.quiver(
            np.zeros_like(dx), np.zeros_like(dy), dx, dy,
            angles="xy", scale_units="xy", scale=1, color="blue", alpha=0.5
        )

        # Elipse 1-sigma em ΔX,ΔY (a sua)
        width, height = 2 * np.sqrt(np.maximum(eigvals, 0))
        ellipse = Ellipse((0, 0), width, height, angle=angulo,
                          edgecolor="red", fc="None", lw=2, label="Elipse (1σ) ΔX,ΔY")
        ax2.add_patch(ellipse)

        ax2.axhline(0, color="black", lw=1)
        ax2.axvline(0, color="black", lw=1)
        ax2.set_aspect("equal")
        ax2.legend()
        ax2.set_title("Distribuição Vetorial com Setas e Elipse")
        ax2.set_xlabel("ΔX")
        ax2.set_ylabel("ΔY")
        st.pyplot(fig2)

    # --- Animação frame a frame (mantida, mas com proteção de índice) ---
    st.subheader("🎮 Passo 3: Animação frame a frame")
    max_frame = len(dx) - 1  # evita estourar índice quando usa dx[frame]
    if max_frame < 1:
        st.info("Poucos pontos para animação.")
        st.stop()

    frame = st.slider("Deslize para visualizar:", 0, max_frame, 1)

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))

    ax3a.plot(x[:frame+1], y[:frame+1], "-", color="black")
    for i in range(frame):
        ax3a.arrow(x[i], y[i], dx[i], dy[i], head_width=2, head_length=2, fc="gray", ec="black")
    ax3a.arrow(x[frame], y[frame], dx[frame], dy[frame], head_width=2, head_length=2, fc="blue", ec="black")

    ax3a.set_title("Trajetória acumulada com vetores")
    ax3a.set_xlabel("X")
    ax3a.set_ylabel("Y")
    ax3a.set_xlim(min(x), max(x))
    ax3a.set_ylim(min(y), max(y))

    ax3b.quiver(
        np.zeros_like(dx[:frame+1]), np.zeros_like(dy[:frame+1]),
        dx[:frame+1], dy[:frame+1],
        angles="xy", scale_units="xy", scale=1, color="black", alpha=0.5
    )

    if frame > 2:
        eixo_maior_f, eixo_menor_f, razao_f, angulo_f, s_index_f, eigvecs_f, eigvals_f = calcular_elipse_inercia(
            dx[:frame+1], dy[:frame+1]
        )
        width_f, height_f = 2 * np.sqrt(np.maximum(eigvals_f, 0))
        ellipse_f = Ellipse((0, 0), width_f, height_f, angle=angulo_f, edgecolor="red", fc="None", lw=2)
        ax3b.add_patch(ellipse_f)
        ax3b.set_title(f"Vetores até o frame {frame}\nS-index parcial: {s_index_f:.2f}")
    else:
        ax3b.set_title("Vetores acumulados")

    ax3b.axhline(0, color="black", lw=1)
    ax3b.axvline(0, color="black", lw=1)
    ax3b.set_xlabel("ΔX")
    ax3b.set_ylabel("ΔY")
    ax3b.set_xlim(-100, 100)
    ax3b.set_ylim(-100, 100)

    st.pyplot(fig3)

else:
    st.info("Aguardando upload de arquivo com colunas: tempo, X, Y...")
