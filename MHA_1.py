import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def fator(b, t):
    denom = -np.expm1(-2*b*t)
    result = np.where(denom == 0, np.inf, 2*np.pi / denom)
    return result

def movimento_amortecido(t, a, b, c, d, offset):
    return a * np.exp(-b * t) * np.cos(c * t - d) + offset

def analise_tabela(file_path):
    try:
        df = pd.read_csv(file_path, sep=",", decimal=",", skiprows=1, header=None)
        df.columns = ["t", "x"]
    except Exception as e:
        print(f"Erro ao ler o arquivo {file_path}: {e}")
        print("Verificar se o arquivo está no formato correto (duas colunas separadas por vírgula, com vírgula como decimal, ignorando a primeira linha).")
        return

    t_data = df["t"].values
    x_data = df["x"].values

    a_initial = (np.max(x_data) - np.min(x_data)) / 2
    offset_initial = np.mean(x_data)

    if len(t_data) > 1 and (t_data[-1] - t_data[0]) > 0:
        c_initial = 2 * np.pi / (t_data[-1] - t_data[0]) * 5 
    else:
        c_initial = 2 * np.pi / 1.0 
    
    b_initial = 0.1 
    d_initial = 0.0 

    initial_guess = [a_initial, b_initial, c_initial, d_initial, offset_initial]
    bounds = ([0, 0, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])

    try:
        param, cov = curve_fit(movimento_amortecido, t_data, x_data, p0=initial_guess, bounds=bounds, maxfev=5000)
    except RuntimeError as e:
        print(f"Erro ao ajustar a curva MHA: {e}")
        print("Tente ajustar as estimativas iniciais ou verificar a qualidade dos dados.")
        return

    a_fit, b_fit, c_fit, d_fit, offset_fit = param
    periodo = 2 * np.pi / c_fit

    print("\n--- Análise Completa da Tabela: " + file_path.split("/")[-1] + " ---")
    print("Coeficientes do MHA ajustados:")
    print(f"  a (Amplitude inicial): {a_fit:.4f}")
    print(f"  b (Coeficiente de amortecimento): {b_fit:.4f}")
    print(f"  c (Frequência angular): {c_fit:.4f} rad/s")
    print(f"  d (Fase): {d_fit:.4f} rad")
    print(f"  Offset (Posição de equilíbrio): {offset_fit:.4f}")
    print(f"  Fator de qualidade: {np.mean(fator(b_fit, periodo)):.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(t_data, x_data, label='Dados Originais', alpha=0.7)
    t_fit = np.linspace(t_data.min(), t_data.max(), 500) 
    x_fit = movimento_amortecido(t_fit, *param)
    plt.plot(t_fit, x_fit, label='Ajuste MHA', color='red', linestyle='--')
    plt.title("Ajuste de MHA para os dados do pêndulo")
    plt.xlabel('Tempo (s)')
    plt.ylabel('Posição')
    plt.legend()
    plt.grid(True)
    plot_filename = file_path.replace("DadosMHA.csv", "analise_total.png")
    plt.savefig(plot_filename)
    print(f"Gráfico do ajuste salvo como {plot_filename}")

if __name__ == "__main__":
    pass


