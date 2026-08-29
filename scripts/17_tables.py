"""Tablas finales del manuscrito generadas desde resultados ejecutados.

La carpeta submission_tables contiene cuatro tablas numeradas en el orden
previsto de aparicion. Cada tabla se exporta en CSV y Markdown. Ningun valor
se introduce a mano salvo los parametros bloqueados del protocolo, cuya
fuente canonica es sensors.py y DECISIONS.md D12 y D17.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from estimhyb import sensors  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results", "primary")
OUT = os.path.join(ROOT, "submission_tables")

MAIN = ["EKF_nom", "EKF_mis", "CovMatch", "SageHusa", "Huber", "Propuesto"]
SCENS = ["S0_nominal", "S1_noise", "S2_bias", "S3_outlier",
         "S4_dropout", "S5_combined"]
SCEN_LABEL = {"S0_nominal": "Sin degradacion", "S1_noise": "Ruido",
              "S2_bias": "Sesgo", "S3_outlier": "Valores atipicos",
              "S4_dropout": "Perdida de datos", "S5_combined": "Combinado"}
ARM_LABEL = {"EKF_nom": "EKF nominal", "EKF_mis": "EKF mal sintonizado",
             "CovMatch": "Correspondencia de covarianza",
             "SageHusa": "Sage Husa", "Huber": "Huber robusto",
             "Propuesto": "Excluir con disparo", "Compensado": "Reparar con disparo",
             "Comp_siempre": "Reparar siempre"}


def interval(x, digits=3):
    """Mediana y rango intercuartilico en una celda compacta."""
    return (f"{np.median(x):.{digits}f} "
            f"[{np.percentile(x, 25):.{digits}f}, "
            f"{np.percentile(x, 75):.{digits}f}]")


def save(df, number, stem):
    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, f"Table_{number:02d}_{stem}")
    df.to_csv(base + ".csv", index=False, encoding="utf-8-sig")
    with open(base + ".md", "w", encoding="utf-8") as fh:
        headers = [str(c) for c in df.columns]
        fh.write("| " + " | ".join(headers) + " |\n")
        fh.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in df.itertuples(index=False, name=None):
            cells = [str(v).replace("|", "\\|") for v in row]
            fh.write("| " + " | ".join(cells) + " |\n")
    print("  escrita", os.path.basename(base) + ".csv y .md")


def table1_protocol():
    rows = [
        ["Duracion por corrida", "120", "s"],
        ["Frecuencia base", "100", "Hz"],
        ["Repeticiones por celda", "40", "repeticiones"],
        ["Severidades", "0, 0.25, 0.50, 0.75, 1.00", "adimensional"],
        ["Factor maximo de ruido", f"{sensors.SEV_NOISE_FACTOR:.0f}", "veces la desviacion nominal"],
        ["Sesgo terminal", f"{sensors.SEV_BIAS_SIGMA:.0f}", "desviaciones nominales"],
        ["Tasa maxima de valores atipicos", f"{sensors.SEV_OUTLIER_RATE:.2f}", "por muestra"],
        ["Magnitud de valor atipico", f"{sensors.SEV_OUTLIER_SIGMA:.0f}", "desviaciones nominales"],
        ["Tasa maxima de inicio de perdida", f"{sensors.SEV_DROPOUT_RATE:.2f}", "por muestra"],
        ["Duracion media maxima de perdida", f"{sensors.SEV_DROPOUT_MEAN:.0f}", "muestras del canal"],
    ]
    return pd.DataFrame(rows, columns=["Parametro", "Valor", "Unidad"])


def table2_estimators():
    rows = [
        ["EKF nominal", "R fija", "escala de Q 0.9976"],
        ["EKF mal sintonizado", "R fija", "escala de Q 0.09976"],
        ["Correspondencia de covarianza", "Escala por traza", "ventana 30, lambda entre 1 y 1000"],
        ["Sage Husa", "Escala por traza", "b 0.98, lambda entre 0.2 y 1000"],
        ["Huber", "Peso equivalente", "kappa 1.345, lambda entre 1 y 1000"],
        ["Dos momentos", "Inflacion y compuerta", "rho 0.5, zona 0.5, alfa NIS 0.01, alfa media 0.001, beta 0.02, tres faltas, permanencia 25"],
        ["Compensacion", "Estado aumentado", "P0 de sesgo 0.01 R, densidad 0.1 sigma, activacion enclavada"],
    ]
    return pd.DataFrame(rows, columns=["Estimador", "Mecanismo", "Parametros"])


def table3_main(df):
    """RMSE de posicion en severidad uno, mediana e intervalo intercuartilico."""
    rows = []
    for arm in MAIN:
        row = {"Estimador": ARM_LABEL[arm]}
        for scen in SCENS:
            sev = 0.0 if scen == "S0_nominal" else 1.0
            x = df[(df.estimador == arm) & (df.escenario == scen)
                   & (df.severidad == sev)].pos_rmse.to_numpy()
            row[SCEN_LABEL[scen]] = interval(x)
        rows.append(row)
    return pd.DataFrame(rows)


def table4_tradeoff(df):
    arms = ["EKF_nom", "Propuesto", "Compensado", "Comp_siempre"]
    base_bias = (df[(df.estimador == "EKF_nom") & (df.escenario == "S2_bias")
                    & (df.severidad == 1.0)]
                 .sort_values("repeticion").pos_rmse.to_numpy())
    rows = []
    for arm in arms:
        nom = df[(df.estimador == arm) & (df.escenario == "S0_nominal")
                 & (df.severidad == 0.0)].pos_rmse.to_numpy()
        bias = (df[(df.estimador == arm) & (df.escenario == "S2_bias")
                   & (df.severidad == 1.0)]
                .sort_values("repeticion").pos_rmse.to_numpy())
        rows.append({
            "Mecanismo": ARM_LABEL[arm],
            "Sin degradacion": interval(nom),
            "Sesgo severidad uno": interval(bias),
            "Mejora frente al nominal": "Referencia" if arm == "EKF_nom"
            else f"{100 * np.median(bias / base_bias - 1.0):.1f} %",
            "Repeticiones con mejora": "Referencia" if arm == "EKF_nom"
            else f"{int((bias < base_bias).sum())} de {bias.size}",
            "Coeficiente de variacion": f"{bias.std(ddof=1) / bias.mean():.3f}",
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(os.path.join(RES, "confirmatory_v2_runs.csv"))
    print("generando cuatro tablas finales")
    save(table1_protocol(), 1, "protocol")
    save(table2_estimators(), 2, "estimators")
    save(table3_main(df), 3, "main_results")
    save(table4_tradeoff(df), 4, "bias_tradeoff")


if __name__ == "__main__":
    main()
