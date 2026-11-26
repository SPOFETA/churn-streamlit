import joblib
import pandas as pd
import streamlit as st

import features   # o modelo salvo chama features.engenharia: o arquivo precisa estar no repositório

st.set_page_config(page_title="Risco de churn — Telco", page_icon="📉")
st.title("Este cliente vai cancelar?")
st.caption("App publicado por: Simão Oliveira")
st.write("Preencha os dados de um cliente da operadora. O modelo foi treinado com 5.634 clientes do "
         "dataset Telco Customer Churn (IBM, Califórnia, EUA) e devolve uma pontuação de risco de cancelamento.")


@st.cache_resource            # carrega o arquivo uma vez só, e não a cada clique
def carregar_modelo():
    return joblib.load("modelo_churn.pkl")    # o pipeline inteiro: features, pré-processamento e o modelo


modelo = carregar_modelo()

LIMIAR = 0.5                  # o mesmo corte do notebook (xgb_final.predict)
SIM_NAO = {"Yes": "Sim", "No": "Não"}

# ── o formulário: um campo para cada coluna bruta que o modelo recebe
st.subheader("Perfil")
c1, c2 = st.columns(2)
with c1:
    genero = st.selectbox("Gênero", ["Female", "Male"], format_func=lambda v: {"Female": "Feminino", "Male": "Masculino"}[v])
    idoso = st.selectbox("Idoso (65 anos ou mais)", ["No", "Yes"], format_func=SIM_NAO.get)
with c2:
    parceiro = st.selectbox("Tem parceiro(a)", ["No", "Yes"], format_func=SIM_NAO.get)
    dependentes = st.selectbox("Tem dependentes", ["No", "Yes"], format_func=SIM_NAO.get)

st.subheader("Contrato e cobrança")
c1, c2 = st.columns(2)
with c1:
    meses = st.slider("Tempo de casa (meses)", 0, 72, 12)
    contrato = st.selectbox("Contrato", ["Month-to-month", "One year", "Two year"],
                            format_func=lambda v: {"Month-to-month": "Mês a mês", "One year": "1 ano",
                                                   "Two year": "2 anos"}[v])
    sem_papel = st.selectbox("Fatura digital (sem papel)", ["Yes", "No"], format_func=SIM_NAO.get)
with c2:
    mensal = st.number_input("Mensalidade (US$)", 18.0, 120.0, 70.0, step=0.5)
    pagamento = st.selectbox("Forma de pagamento",
                             ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    # o valor inicial é uma estimativa (meses × mensalidade); dá para corrigir
    total = st.number_input("Total já faturado (US$)", 0.0, 9000.0, float(round(meses * mensal, 2)), step=10.0,
                            help="Estimativa inicial: tempo de casa × mensalidade. Corrija se souber o valor real.")

st.subheader("Serviços")
c1, c2 = st.columns(2)
with c1:
    telefone = st.selectbox("Serviço de telefone", ["Yes", "No"], format_func=SIM_NAO.get)
    internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"],
                            format_func=lambda v: {"DSL": "DSL", "Fiber optic": "Fibra óptica", "No": "Sem internet"}[v])

# Quem não tem telefone ou internet não pode ter os serviços que dependem deles: no dataset de treino
# essas respostas são "No phone service" / "No internet service", e é assim que o modelo as conhece.
linhas = telefone == "Yes"
com_internet = internet != "No"
with c2:
    if linhas:
        multiplas = st.selectbox("Várias linhas", ["No", "Yes"], format_func=SIM_NAO.get)
    else:
        multiplas = "No phone service"
    DEPENDEM_DA_INTERNET = {"Online Security": "Segurança online", "Online Backup": "Backup online",
                            "Device Protection": "Proteção de dispositivo", "Tech Support": "Suporte técnico",
                            "Streaming TV": "Streaming de TV", "Streaming Movies": "Streaming de filmes"}
    extras = {}
    for coluna, rotulo in DEPENDEM_DA_INTERNET.items():
        if com_internet:
            extras[coluna] = st.selectbox(rotulo, ["No", "Yes"], format_func=SIM_NAO.get)
        else:
            extras[coluna] = "No internet service"
if not com_internet:
    st.caption("Sem internet, os serviços que dependem dela ficam como “No internet service”, como no treino.")

# ── uma linha BRUTA, com os mesmos nomes de coluna do treino; o pipeline faz o resto
cliente = pd.DataFrame([{
    "Gender": genero, "Senior Citizen": idoso, "Partner": parceiro, "Dependents": dependentes,
    "Tenure Months": meses, "Phone Service": telefone, "Multiple Lines": multiplas,
    "Internet Service": internet, **extras,
    "Contract": contrato, "Paperless Billing": sem_papel, "Payment Method": pagamento,
    "Monthly Charges": mensal, "Total Charges": total,
}])
pontuacao = modelo.predict_proba(cliente)[0, 1]
risco_alto = pontuacao >= LIMIAR

st.divider()
m1, m2 = st.columns(2)
m1.metric("Pontuação de risco do modelo", f"{pontuacao:.0%}")
m2.metric("Classificação", "Risco alto" if risco_alto else "Risco baixo")
if risco_alto:
    st.warning("O modelo classifica este cliente como de risco alto de cancelamento: vale uma ação de retenção.")
else:
    st.success("O modelo classifica este cliente como de risco baixo de cancelamento.")
st.caption("A pontuação não é a probabilidade real de cancelar: o modelo foi treinado com peso extra nos "
           "clientes que cancelam (scale_pos_weight ≈ 2,77), então ela fica inflada. É uma estimativa para "
           "apoiar a equipe de retenção, não uma decisão sobre o cliente. O app não guarda o que você digita.")

with st.expander("O que o modelo recebe"):
    st.write("A linha bruta que o app monta e entrega ao `predict`; o pipeline cria as 4 features derivadas sozinho:")
    st.dataframe(cliente, hide_index=True)
    st.write("As features derivadas, como o pipeline as calcula:")
    st.dataframe(features.engenharia(cliente)[["Tenure_Cohort", "Digital_Security_Index",
                                               "Fiber_Without_TechSupport", "Total_Services_Count"]],
                 hide_index=True)
