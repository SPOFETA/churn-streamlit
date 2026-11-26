# Este cliente vai cancelar?

App em Streamlit que estima o risco churn de um cliente de operadora de telecomunicações.
Necessário preencher o perfil, o contrato e os serviços do cliente, e o modelo devolve uma pontuação de risco e uma
classificação (risco alto ou baixo).

link: https://churn-telco-inf1032.streamlit.app/

Projeto em grupo de INF1032 (Introdução à Ciência de Dados, PUC-Rio), Turma 3WB, Grupo 02.

## Os arquivos

| arquivo | o que faz |
|---|---|
| `app.py` | a tela: carrega o modelo, mostra o formulário e calcula a pontuação |
| `features.py` | as funções de engenharia de features; o modelo salvo chama `features.engenharia` |
| `modelo_churn.pkl` | o pipeline treinado (features, pré-processamento e o XGBoost tunado) |
| `requirements.txt` | as bibliotecas que o Streamlit instala antes de rodar o app, com as versões que geraram o `.pkl` |
| `README.md` | esta página |

## O modelo

É o pipeline do projeto: o app entrega ao modelo a linha **bruta** do cliente, e o pipeline cria as
features derivadas (`Tenure_Cohort`, `Digital_Security_Index`, `Fiber_Without_TechSupport` e
`Total_Services_Count`), padroniza as numéricas, codifica as categóricas e classifica com um `XGBClassifier`
ajustado por `RandomizedSearchCV` (10 sorteios, validação cruzada de 5 partes, métrica F1 da classe churn).
Os parâmetros escolhidos foram `max_depth=5`, `learning_rate=0,05`, `n_estimators=100` e
`scale_pos_weight` ≈ 2,77, para compensar o desbalanceamento (~26,5% de churn).
Foi treinado com 5.634 clientes do dataset Telco Customer Churn (IBM, Califórnia, EUA).

No conjunto de teste do notebook (1.409 clientes, corte em 0,5), a classe de churn teve F1 de 0,631,
recall de 0,791 (296 dos 374 cancelamentos) e precisão de 0,525. Na validação cruzada do treino, o F1 foi 0,647.

## Limites e cuidados

- **A pontuação não é a probabilidade real de cancelar.** Como o modelo foi treinado com peso extra nos
  clientes que cancelam, a pontuação fica inflada. Ela serve para ordenar e para o corte em 0,5, não para
  dizer "este cliente tem X% de chance de sair".
- **É uma estimativa.** Serve para apoiar a equipe de retenção; nenhuma decisão sobre um
  cliente deve ser tomada só por ela (LGPD, art. 20: quem é afetado por uma decisão automatizada pode pedir
  a revisão dela).
- **O app não guarda o que é digitado.** O repositório não tem o dataset de treino, só o modelo.
- **Dados de uma só região e época.** O modelo conhece clientes da Califórnia, num dataset de exemplo da
  IBM; fora dessa faixa, a previsão é uma extrapolação. Não há monitoramento de drift neste app.
- Quem não tem telefone ou internet não pode ter os serviços que dependem deles, e o formulário trata isso
  como no treino (`No phone service` e `No internet service`).

## Rodar no seu computador

```bash
pip install -r requirements.txt
streamlit run app.py
```
