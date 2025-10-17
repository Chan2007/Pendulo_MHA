import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from MHA_1 import analise_tabela
from MHA_1 import fator
from MHA_1 import movimento_amortecido

def fit_desvio(t_data, x_data):
    if len(t_data) < 5: # Precisa de pelo menos 5 pontos
        return np.inf, None

    a_initial = (np.max(x_data) - np.min(x_data)) / 2
    if a_initial == 0: 
        a_initial = 1

    offset_initial = np.mean(x_data)

    # Estimativa de frequência
    # Tentativa de estimar o período a partir dos cruzamentos com o eixo x
    zero_crossings = np.where(np.diff(np.sign(x_data - offset_initial)))[0]
    if len(zero_crossings) > 1:
        # Calcular a média dos intervalos entre os cruzamentos de zero no eixo x
        # Multiplicar por 2 para obter o período completo (ida e volta)
        estimativa_periodo = np.mean(np.diff(t_data)[zero_crossings[1:] - zero_crossings[:-1]]) * 2
        if estimativa_periodo > 0:
            c_initial = 2 * np.pi / estimativa_periodo
        else:
            c_initial = 2 * np.pi / (t_data[-1] - t_data[0]) if (t_data[-1] - t_data[0]) > 0 else 2*np.pi # Padrão se período não for detectado
    else:
        c_initial = 2 * np.pi / (t_data[-1] - t_data[0]) if (t_data[-1] - t_data[0]) > 0 else 2*np.pi # Padrão se não houver cruzamentos

    b_initial = 0.1
    d_initial = 0.0

    initial_guess = [a_initial, b_initial, c_initial, d_initial, offset_initial]
    bounds = ([0, 0, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])

    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            param, cov = curve_fit(movimento_amortecido, t_data, x_data, p0=initial_guess, bounds=bounds, maxfev=10000)
        ideal = movimento_amortecido(t_data, *param)
        desvios = x_data - ideal
        desvio_padrao = np.std(desvios)
        return desvio_padrao, param
    except RuntimeError:
        return np.inf, None

def analise_modulada(file_path, num_points=30):
    try:
        # Leitura do arquivo CSV
        try:
            df = pd.read_csv(file_path, sep=",", decimal=",", skiprows=1, header=None, names=["t", "x"])
        except:
            df = pd.read_csv(file_path, sep=",", decimal=",", skiprows=1, header=None, names=["t", "x"])

    except Exception as e:
        print(f"Erro ao ler o arquivo {file_path}: {e}")
        return

    min_desvio_padrao = np.inf
    best_params = None
    best_data_subset = None
    best_start_index = -1

    # Certificar-se de que há dados suficientes para formar um subconjunto de num_points
    if len(df) < num_points:
        print(f"O arquivo {file_path} tem menos de {num_points} pontos. Não é possível analisar.")
        return

    for i in range(len(df) - num_points + 1):
        subset_df = df.iloc[i:i + num_points]
        t_subset = subset_df["t"].values
        x_subset = subset_df["x"].values

        desvio_padrao, params = fit_desvio(t_subset, x_subset)

        if desvio_padrao < min_desvio_padrao:
            min_desvio_padrao = desvio_padrao
            best_params = params
            best_data_subset = subset_df
            best_start_index = i

    if best_params is not None:
        periodo = 2 * np.pi / best_params[2]
        print("\n--- Análise Modulada: " + file_path.split("/")[-1] + " ---")
        print(f"Melhor desvio padrão encontrado: {min_desvio_padrao:.4f}")
        print(f"Início do subconjunto: índice {best_start_index}")
        print("Coeficientes do MHA ajustados:")
        print(f"  a: {best_params[0]:.4f}")
        print(f"  b: {best_params[1]:.4f}")
        print(f"  c: {best_params[2]:.4f} rad/s")
        print(f"  d: {best_params[3]:.4f} rad")
        print(f"  Offset: {best_params[4]:.4f}")
        print(f"  Fator de qualidade: {np.mean(fator(best_params[1], periodo)):.4f}")

        output_csv_filename = file_path.replace(".csv", "_planilha.csv")
        best_data_subset.to_csv(output_csv_filename, index=False)
        print(f"Planilha salva em: {output_csv_filename}")

        plt.figure(figsize=(10, 6))
        plt.plot(best_data_subset["t"], best_data_subset["x"], "o", label="Dados Selecionados")
        t_fit = np.linspace(best_data_subset["t"].min(), best_data_subset["t"].max(), 200)
        x_fit = movimento_amortecido(t_fit, *best_params)
        plt.plot(t_fit, x_fit, "r--", label="Ajuste MHA")
        plt.title("Ajuste MHA para " + file_path.split("/")[-1] + " (os 30 \"melhores\" pontos)")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Posição")
        plt.legend()
        plt.grid(True)
        plot_filename = file_path.replace(".csv", "grafico_pontos.png")
        plt.savefig(plot_filename)
        print(f"Gráfico do ajuste salvo em: {plot_filename}")

    else:
        print(f"Não foi possível encontrar um ajuste para {file_path}. Verifique os dados.")

if __name__ == "__main__":
    # Caminho para o arquivo de dados do usuário
    data_file = "UTFPR/FISICA/PENDULO/DadosMHA.csv"
    analise_tabela(data_file)
    analise_modulada(data_file)  # Com o número de iterações e operações, essa função demora muito para produzir os resultados (~ 10 minutos)

