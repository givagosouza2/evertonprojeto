import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import math
import io

st.set_page_config(layout="wide")
st.title("Análise Vetorial + Elipse 95% + Intervalos e Sequência de Toques")

uploaded_file = st.file_uploader(
    "📄 Carregue um arquivo .txt ou .csv com colunas: Tempo, X, Y, ...",
    type=["txt", "csv"]
)

def _to_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def ler_arquivo(uploaded_file):
    raw = uploaded_file.read()
    text = raw.decode("utf-8-sig", errors="ignore")

    # 1) tenta TAB, que é o caso do arquivo anexado
    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep="\t",
            engine="python",
            skipinitialspace=True
        )
        if df.shape[1] >= 3:
            df.columns = [str(c).strip() for c in df.columns]
            return df
    except Exception:
        pass

    # 2) tenta separador automático mais amplo
    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=r"[;,]|\s{2,}",
            engine="python"
        )
        if df.shape[1] >= 3:
            df.columns = [str(c).strip() for c in df.columns]
            return df
    except Exception:
        pass

    # 3) fallback
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def selecionar_colunas(df):
    cols = {str(c).strip().lower(): c for c in df.columns}

    t_col = cols.get("time", None)
    x_col = cols.get("x", None)
    y_col = cols.get("y", None)

    if t_col is None:
        t_col = df.columns[0]
    if x_col is None:
        x_col = df.columns[1]
    if y_col is None:
        y_col = df.columns[2]

    return t_col, x_col, y_col

def infer_time_to_seconds(t):
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if len(dt) == 0:
        return t, "desconhecida"
    med = np.median(dt)
    if med > 20:
        return t / 1000.0, "ms→s"
    return t, "s"

def calcular_elipse_inercia(x, y):
    X = np.vstack([x, y])
    cov = np.cov(X)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvals = np.maximum(eigvals, 0)

    eixo_maior = np.sqrt(eigvals[0])
    eixo_menor = np.sqrt(eigvals[1])
    razao = eixo_maior / eixo_menor if eixo_menor != 0 else np.inf
    angulo = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    s_index = razao
    return eixo_maior, eixo_menor, razao, angulo, s_index, eigvecs, eigvals

def elipse_95_xy(x, y):
    chi2_95_df2 = 5.991464547107979
    mu_x, mu_y = np.mean(x), np.mean(y)
    _, _, _, ang, _, eigvecs, eigvals = calcular_elipse_inercia(x, y)
    a = np.sqrt(chi2_95_df2 * max(eigvals[0], 0))
    b = np.sqrt(chi2_95_df2 * max(eigvals[1], 0))
    area = math.pi * a * b
    return (mu_x, mu_y), a, b, ang, area, eigvals, eigvecs

if uploaded_file:
    try:
        df = ler_arquivo(uploaded_file)

        st.write("Colunas detectadas:", list(df.columns))
        st.dataframe(df.head())

        if df.shape[1] < 3:
            st.error("O arquivo deve conter pelo menos três colunas: tempo, X e Y.")
            st.stop()

        t_col, x_col, y_col = selecionar_colunas(df)

        t_raw = _to_numeric(df[t_col])
        x_raw = _to_numeric(df[x_col])
        y_raw = _to_numeric(df[y_col])

        valid = (~t_raw.isna()) & (~x_raw.isna()) & (~y_raw.isna())

        dfv = pd.DataFrame({
            "t": t_raw[valid],
            "x": x_raw[valid],
            "y": y_raw[valid]
        }).reset_index(drop=True)

        st.write(f"Pontos válidos após limpeza: {len(dfv)}")

        if len(dfv) < 3:
            st.error("Poucos pontos válidos após limpeza. Verifique o arquivo.")
            st.stop()

        t = dfv["t"].to_numpy()
        x = dfv["x"].to_numpy()
        y = dfv["y"].to_numpy()

        dx = np.diff(x)
        dy = np.diff(y)

        if len(dx) < 2:
            st.error("O arquivo não possui dados suficientes para calcular vetores.")
            st.stop()

        t_sec, time_unit = infer_time_to_seconds(t)

        n_toques = len(x)

        dt = np.diff(t_sec)
        dt = dt[np.isfinite(dt)]
        dt = dt[dt > 0]
        intervalo_medio = float(np.mean(dt)) if len(dt) else np.nan

        dr = np.sqrt(dx**2 + dy**2)
        soma_resultante_espacial = float(np.nansum(dr))

        (cx, cy), a95, b95, ang95, area95, eigvals_xy, eigvecs_xy = elipse_95_xy(x, y)
        eixo_maior, eixo_menor, razao, angulo, s_index, eigvecs, eigvals = calcular_elipse_inercia(dx, dy)

        st.subheader("📊 Métricas principais")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nº de toques", f"{n_toques}")
        c2.metric("Intervalo médio entre toques (s)", f"{intervalo_medio:.4f}" if np.isfinite(intervalo_medio) else "—")
        c3.metric("∑ resultante espacial (∑‖Δr‖)", f"{soma_resultante_espacial:.4f}")
        c4.metric("Área elipse 95% (X,Y)", f"{area95:.4f}")

        st.caption(f"Colunas usadas: tempo={t_col}, X={x_col}, Y={y_col}")
        st.caption(f"Tempo interpretado como: **{time_unit}**")

        st.subheader("⏱ Intervalo entre toques e sequência dos toques")

        colA, colB = st.columns(2)

        with colA:
            if len(dt) > 0:
                fig_dt, ax_dt = plt.subplots(figsize=(8, 4))
                ax_dt.plot(t_sec[0:-1], dt, marker="o")
                ax_dt.set_xlabel("Time (s)")
                ax_dt.set_ylabel("Intertap interval (s)")
                ax_dt.grid(True, alpha=0.3)
                st.pyplot(fig_dt)

        with colB:
            idx = np.arange(1, n_toques + 1)
            fig_seq, ax_seq = plt.subplots(figsize=(8, 4))
            ax_seq.plot(idx, x, marker="o", label="X")
            ax_seq.plot(idx, y, marker="o", label="Y")
            ax_seq.set_title("X e Y ao longo da sequência")
            ax_seq.set_xlabel("Índice do toque")
            ax_seq.set_ylabel("Coordenada")
            ax_seq.grid(True, alpha=0.3)
            ax_seq.legend()
            st.pyplot(fig_seq)

        if len(dr) > 0:
            fig_dr, ax_dr = plt.subplots(figsize=(12, 3.5))
            ax_dr.plot(np.arange(1, len(dr) + 1), dr, marker="o")
            ax_dr.set_title("Distância entre toques consecutivos (‖Δr‖)")
            ax_dr.set_xlabel("Índice do deslocamento")
            ax_dr.set_ylabel("‖Δr‖")
            ax_dr.grid(True, alpha=0.3)
            st.pyplot(fig_dr)

        st.subheader("📍 Posições (X,Y) + Elipse 95% / Vetores (ΔX,ΔY) + Elipse")

        col1, col2 = st.columns([0.5, 1])

        with col1:
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            ax1.plot(x, y, "o-", alpha=0.6, label="Trajetória")

            rect = Rectangle((0, 0), 1440, 2730, linewidth=1, edgecolor="black", facecolor="none")
            ax1.add_patch(rect)

            ellipse_xy = Ellipse(
                (cx, cy),
                width=2 * a95,
                height=2 * b95,
                angle=ang95,
                edgecolor="red",
                fc="None",
                lw=2,
                label="Elipse 95%"
            )
            ax1.add_patch(ellipse_xy)

            ax1.set_aspect("equal", adjustable="box")
            ax1.set_xlim(0, 1440)
            ax1.set_ylim(0, 2730)
            ax1.set_title("Coordenadas espaciais (X, Y) + Elipse 95%")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.legend()
            st.pyplot(fig1, clear_figure=True)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            ax2.scatter(dx, dy, alpha=0.6, label="Vetores de deslocamento")
            ax2.quiver(
                np.zeros_like(dx), np.zeros_like(dy), dx, dy,
                angles="xy", scale_units="xy", scale=1, color="blue", alpha=0.5
            )

            width, height = 2 * np.sqrt(np.maximum(eigvals, 0))
            ellipse = Ellipse(
                (0, 0), width, height, angle=angulo,
                edgecolor="red", fc="None", lw=2, label="Elipse (1σ) ΔX,ΔY"
            )
            ax2.add_patch(ellipse)

            ax2.axhline(0, color="black", lw=1)
            ax2.axvline(0, color="black", lw=1)
            ax2.set_aspect("equal")
            ax2.legend()
            ax2.set_title(f"Distribuição Vetorial com Setas e Elipse | S-index={s_index:.2f}")
            ax2.set_xlabel("ΔX")
            ax2.set_ylabel("ΔY")
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"Erro ao abrir/processar o arquivo: {e}")

else:
    st.info("Aguardando upload de arquivo com colunas: tempo, X, Y...")
