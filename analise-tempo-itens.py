# -*- coding: utf-8 -*-
import pandas as pd
from pulp import *
import math
import time 

# ============================================================
# 0. PARÂMETROS DO MODELO
# ============================================================
CAPACIDADE_DIA    = 8.22
SETUP_POR_GRUPO   = 0.1667
LIMITE_SETUPS_DIA = 5
DIAS_COBERTURA    = 4 

TEMPOS_PARA_TESTAR = [5, 30, 50, 120]
log_por_item = []

DATA_MPS = pd.to_datetime("2026-02-02")
DATA_FIM_MPS = pd.to_datetime("2026-02-28") 
DATA_FIM_EXT = pd.to_datetime("2026-03-31") 

arquivo = "dados_planejamento.xlsx"

# ============================================================
# 1. DADOS DE ENTRADA E PRÉ-PROCESSAMENTO
# ============================================================
item_df = pd.read_excel(arquivo, sheet_name="item")
demanda_df = pd.read_excel(arquivo, sheet_name="demanda")
estoque_df = pd.read_excel(arquivo, sheet_name="estoque_inicial")
feriados_df = pd.read_excel(arquivo, sheet_name="feriados")

tempo_unitario = dict(zip(item_df["item"], item_df["a_i"]))
lote = dict(zip(item_df["item"], item_df["L_i"]))
grupo_item = dict(zip(item_df["item"], item_df["grupo"]))
P, G = item_df["item"].tolist(), item_df["grupo"].unique().tolist()

tempo_lote = {i: tempo_unitario[i] * lote[i] for i in P}
big_m_item = {
    i: math.floor((CAPACIDADE_DIA - SETUP_POR_GRUPO) / tempo_lote[i]) if tempo_lote[i] > 0 else 0 
    for i in P
}

feriados = set(pd.to_datetime(feriados_df["data"], dayfirst=True))
def eh_dia_util(d): return d.weekday() < 5 and d not in feriados

T_planejamento = [d for d in pd.date_range(DATA_MPS, DATA_FIM_MPS) if eh_dia_util(d)]
T_full = [d for d in pd.date_range(DATA_MPS, DATA_FIM_EXT) if eh_dia_util(d)]
idx_full = {d: i for i, d in enumerate(T_full)}

def mapear_para_util_anterior(data_alvo):
    d = pd.to_datetime(data_alvo)
    while not eh_dia_util(d) and d > DATA_MPS: d -= pd.Timedelta(days=1)
    return d

demanda_ajustada = {}
for _, r in demanda_df.iterrows():
    data_original = pd.to_datetime(r["data"])
    data_destino = data_original if eh_dia_util(data_original) else mapear_para_util_anterior(data_original)
    chave = (r["item"], data_destino)
    demanda_ajustada[chave] = demanda_ajustada.get(chave, 0) + r["quantidade"]

def get_demanda(i, t): return demanda_ajustada.get((i, t), 0)
def calcular_alvo_regra(i, t, n_dias):
    idx = idx_full[t]
    futuro = T_full[idx + 1 : idx + 1 + n_dias]
    soma_f = sum(get_demanda(i, d) for d in futuro)
    li = max(lote[i], 1)
    return math.ceil(soma_f / li) * li

# ============================================================
# 2. RESOLUÇÃO 
# ============================================================
for tempo_da_vez in TEMPOS_PARA_TESTAR:
    print(f"\n>>> PROCESSANDO TEMPO: {tempo_da_vez}s")
    
    estoque_hoje = {i: 0.0 for i in P}
    for _, r in estoque_df.iterrows():
        if r["item"] in estoque_hoje: estoque_hoje[r["item"]] = float(r["quantidade"])
    atraso_hoje = {i: 0.0 for i in P}

    for t_hoje in T_planejamento:
        idx_h = idx_full[t_hoje]
        horizonte_local = T_full[idx_h : idx_h + DIAS_COBERTURA + 1]
        
        prob = LpProblem(f"Plan_{t_hoje.strftime('%Y%m%d')}", LpMinimize)
        k = LpVariable.dicts("k", (P, horizonte_local), lowBound=0, cat=LpInteger)
        z = LpVariable.dicts("z", (G, horizonte_local), cat=LpBinary)
        I = LpVariable.dicts("I", (P, horizonte_local), lowBound=0)
        B = LpVariable.dicts("B", (P, horizonte_local), lowBound=0)
        S = LpVariable.dicts("S", (P, horizonte_local), lowBound=0)

        for idx_l, t_l in enumerate(horizonte_local):
            prob += lpSum(tempo_lote[i]*k[i][t_l] for i in P) + lpSum(SETUP_POR_GRUPO*z[g][t_l] for g in G) <= CAPACIDADE_DIA
            prob += lpSum(z[g][t_l] for g in G) <= LIMITE_SETUPS_DIA
            
            for i in P:
                prob += k[i][t_l] <= big_m_item[i] * z[grupo_item[i]][t_l]
                I_ant, B_ant = (estoque_hoje[i], atraso_hoje[i]) if idx_l == 0 else (I[i][horizonte_local[idx_l-1]], B[i][horizonte_local[idx_l-1]])
                prob += (I[i][t_l] - B[i][t_l] == I_ant - B_ant + lote[i]*k[i][t_l] - get_demanda(i, t_l))
                
                alvo = calcular_alvo_regra(i, t_l, DIAS_COBERTURA)
                prob += I[i][t_l] + S[i][t_l] >= alvo
                
                if idx_l == 0:
                    teto = max(alvo, estoque_hoje[i])
                    prob += I_ant + lote[i]*k[i][t_l] - get_demanda(i, t_l) <= teto + B[i][t_l]

        # FASE 1: ATRASO
        prob.setObjective(lpSum(B[i][t] / max(1, lote[i]) for i in P for t in horizonte_local))
        prob.solve(PULP_CBC_CMD(msg=0, gapRel=0.0))
        pecas_atraso_total = sum(value(B[i][t]) for i in P for t in horizonte_local)

        # FASE 2: ESTOQUE (Folga de 0.01) com parada por TEMPO
        prob += lpSum(B[i][t] for i in P for t in horizonte_local) <= pecas_atraso_total + 0.01
        prob.setObjective(lpSum(S[i][t] / max(1, lote[i]) for i in P for t in horizonte_local))
        status_f2 = prob.solve(PULP_CBC_CMD(msg=0, timeLimit=tempo_da_vez))

        # EXTRAÇÃO POR ITEM COM FILTRO DE TOLERÂNCIA
        for i in P:
            v1_raw = sum(value(B[i][t]) / max(1, lote[i]) for t in horizonte_local)
            v2_raw = sum(value(S[i][t]) / max(1, lote[i]) for t in horizonte_local)
            
            z1_item = v1_raw if v1_raw > 0.001 else 0.0
            z2_item = v2_raw if v2_raw > 0.001 else 0.0
            
            prod_lotes = value(k[i][t_hoje]) or 0
            
            log_por_item.append({
                "Cenário (tempo)": tempo_da_vez,
                "Data do planejamento": t_hoje.strftime("%d/%m/%Y"),
                "Item": i,
                "Z1: Atraso no horizonte (lotes)": round(z1_item, 4),
                "Z2: Desvio do estoque alvo (lotes)": round(z2_item, 4),
                "Plano de Produção (lotes)": prod_lotes,
                "Plano de Produção (peças)": prod_lotes * lote[i],
                "Status da convergência": LpStatus[status_f2]
            })
            
            estoque_hoje[i] = round(value(I[i][t_hoje]), 6)
            atraso_hoje[i] = round(value(B[i][t_hoje]), 6)
        
        print(f"  [OK] Dia {t_hoje.strftime('%d/%m')} Processado.")

# ============================================================
# 3. EXPORTAÇÃO
# ============================================================
pd.DataFrame(log_por_item).to_excel("analise_tempo_itens.xlsx", index=False)
print(f"\n✔ Sucesso! Arquivo gerado: analise_tempo_itens.xlsx")
