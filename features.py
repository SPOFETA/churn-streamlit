"""Engenharia de features do projeto de churn (INF1032, Grupo 02).

Estas funções são as mesmas do notebook. Ficam num arquivo à parte porque o modelo salvo
(modelo_churn.pkl) chama `features.engenharia`: o app recebe a linha BRUTA do cliente e o
pipeline cria as features sozinho, do mesmo jeito que no treino (sem training-serving skew).
"""
import numpy as np


def gerar_novas_features(dataframe):
    df_new = dataframe.copy()

    # 1. Tenure_Cohort (Mudança de Forma)
    conditions = [
        (df_new['Tenure Months'] <= 6),
        (df_new['Tenure Months'] > 6) & (df_new['Tenure Months'] <= 24),
        (df_new['Tenure Months'] > 24) & (df_new['Tenure Months'] <= 48),
        (df_new['Tenure Months'] > 48)
    ]
    choices = ['Bronze (<6m)', 'Prata (6-24m)', 'Ouro (24-48m)', 'Platina (>48m)']
    df_new['Tenure_Cohort'] = np.select(conditions, choices, default='Bronze (<6m)')

    # 2. Digital_Security_Index (Agregação)
    servicos_seguranca = ['Online Security', 'Online Backup', 'Device Protection', 'Tech Support']
    df_new['Digital_Security_Index'] = 0
    for s in servicos_seguranca:
        df_new['Digital_Security_Index'] += df_new[s].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

    # 3. Fiber_Without_TechSupport (Interação)
    df_new['Fiber_Without_TechSupport'] = ((df_new['Internet Service'] == 'Fiber optic') &
                                            (df_new['Tech Support'] != 'Yes')).astype(int)

    return df_new


def criar_indice_total_servicos(dataframe):
    df_temp = dataframe.copy()

    colunas_servicos = [
        'Phone Service', 'Multiple Lines', 'Internet Service',
        'Online Security', 'Online Backup', 'Device Protection',
        'Tech Support', 'Streaming TV', 'Streaming Movies'
    ]

    def contar_servicos(row):
        contagem = 0
        for col in colunas_servicos:
            val = str(row[col]).strip().lower()
            if val not in ['no', 'no internet service', 'no phone service', 'nan', 'none']:
                contagem += 1
        return contagem

    df_temp['Total_Services_Count'] = df_temp.apply(contar_servicos, axis=1)
    return df_temp


def engenharia(dataframe):
    """A sequência do notebook: gerar_novas_features e depois criar_indice_total_servicos."""
    return criar_indice_total_servicos(gerar_novas_features(dataframe))
