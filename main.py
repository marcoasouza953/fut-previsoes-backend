import os
import json
import requests
import pandas as pd
import numpy as np
import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------------------------
# 1. CONFIGURAÇÕES DINÂMICAS 📅
# -------------------------------------------------------------------------
# Tenta pegar a chave do ambiente (GitHub Actions), senão usa a string direta
# IMPORTANTE: Se fores rodar no PC, substitua "SUA_API_KEY_AQUI" pela tua chave real.
API_KEY = os.environ.get("FOOTBALL_API_KEY", "SUA_API_KEY_AQUI") 

# --- VERIFICAÇÃO DE SEGURANÇA (NOVO) ---
if API_KEY == "SUA_API_KEY_AQUI":
    print("\n" + "="*60)
    print("❌ ERRO: API KEY NÃO CONFIGURADA!")
    print("   Você precisa substituir 'SUA_API_KEY_AQUI' na linha 16")
    print("   pela sua chave verdadeira da API-Sports.")
    print("="*60 + "\n")
    exit() # Para o código aqui para você não perder tempo rodando sem chave
# ---------------------------------------

LEAGUE_ID = "71"  # Brasileirão Série A

# Pega o ano atual do sistema automaticamente
hoje = datetime.datetime.now()
ANO_ATUAL = hoje.year

# Define as temporadas dinamicamente
SEASON_CURRENT = str(ANO_ATUAL)       # Ex: "2025"
SEASON_TRAIN_HISTORIC = str(ANO_ATUAL - 1) # Ex: "2024"

print(f"⚙️ Configuração Automática: Treinando com {SEASON_TRAIN_HISTORIC} (Histórico) + {SEASON_CURRENT} (Realizados)")

# -------------------------------------------------------------------------
# 2. CONEXÃO COM O FIREBASE
# -------------------------------------------------------------------------
print("1. Conectando ao Firebase...")
if not firebase_admin._apps:
    # 1. Tenta pegar do Segredo do GitHub (Variável de Ambiente)
    firebase_creds_str = os.environ.get("FIREBASE_CREDENTIALS")
    
    if firebase_creds_str:
        print("☁️ Usando credenciais da Nuvem (GitHub Secret)")
        cred = credentials.Certificate(json.loads(firebase_creds_str))
        firebase_admin.initialize_app(cred)
    else:
        # 2. Fallback: Tenta pegar arquivo local (para rodar no PC/Colab)
        local_key_path = "firebase_key.json" 
        if os.path.exists(local_key_path):
            print("💻 Usando arquivo local")
            cred = credentials.Certificate(local_key_path)
            firebase_admin.initialize_app(cred)
        else:
            print("⚠️ AVISO: Sem credenciais. O salvamento falhará.")

if firebase_admin._apps:
    db = firestore.client()
    print("✅ Conectado ao banco de dados!")

# -------------------------------------------------------------------------
# 3. LÓGICA DE INTELIGÊNCIA ARTIFICIAL
# -------------------------------------------------------------------------

def coletar_jogos_realizados(season):
    """Baixa apenas jogos TERMINADOS (FT) de uma temporada."""
    print(f"   -> Baixando histórico de {season}...")
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {'x-apisports-key': API_KEY}
    params = {"league": LEAGUE_ID, "season": season, "status": "FT"} # Apenas terminados
    
    try:
        resp = requests.get(url, headers=headers, params=params)
        data = resp.json()
    except Exception as e:
        print(f"❌ Erro na API: {e}")
        return pd.DataFrame()
    
    jogos = []
    if 'response' in data:
        for item in data['response']:
            home = item['goals']['home']
            away = item['goals']['away']
            # Proteção contra dados nulos
            if home is None or away is None: continue
            
            result = 1 if home > away else (2 if away > home else 0)
            
            jogos.append({
                'date': item['fixture']['date'],
                'home_team': item['teams']['home']['name'],
                'away_team': item['teams']['away']['name'],
                'home_goals': home,
                'away_goals': away,
                'result': result
            })
    
    df = pd.DataFrame(jogos)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    return df

def engenharia_de_features(df):
    """
    Reconstrói a linha do tempo do campeonato para calcular a 'força' 
    dos times em cada momento.
    """
    stats = {}
    all_teams = set(df['home_team']).union(set(df['away_team']))
    
    # Inicializa estatísticas zeradas
    for team in all_teams:
        stats[team] = {
            'points': 0, 'games': 0, 
            'goals_scored': 0, 'goals_conceded': 0,
            'last_5': []
        }

    features_list = []

    # Processa jogo a jogo na ordem cronológica
    for index, row in df.iterrows():
        h = row['home_team']
        a = row['away_team']
        
        # --- 1. PREPARAÇÃO (Estado ANTES do jogo começar) ---
        def get_avg(team, metric):
            if stats[team]['games'] == 0: return 0
            return stats[team][metric] / stats[team]['games']

        def get_form(team):
            return sum(stats[team]['last_5'])

        features = {
            'home_points': stats[h]['points'],
            'away_points': stats[a]['points'],
            'points_diff': stats[h]['points'] - stats[a]['points'],
            'home_form': get_form(h),
            'away_form': get_form(a),
            'home_attack': get_avg(h, 'goals_scored'),
            'away_defense': get_avg(a, 'goals_conceded'),
            'away_attack': get_avg(a, 'goals_scored'),
            'home_defense': get_avg(h, 'goals_conceded'),
        }
        features_list.append(features)

        # --- 2. ATUALIZAÇÃO (Estado DEPOIS do jogo acabar) ---
        stats[h]['games'] += 1
        stats[a]['games'] += 1
        stats[h]['goals_scored'] += row['home_goals']
        stats[h]['goals_conceded'] += row['away_goals']
        stats[a]['goals_scored'] += row['away_goals']
        stats[a]['goals_conceded'] += row['home_goals']
        
        pts_h, pts_a = 0, 0
        if row['result'] == 1: pts_h = 3
        elif row['result'] == 2: pts_a = 3
        else: pts_h, pts_a = 1, 1
            
        stats[h]['points'] += pts_h
        stats[a]['points'] += pts_a
        
        # Atualiza fila de forma (mantém apenas últimos 5)
        stats[h]['last_5'].append(pts_h)
        if len(stats[h]['last_5']) > 5: stats[h]['last_5'].pop(0)
        stats[a]['last_5'].append(pts_a)
        if len(stats[a]['last_5']) > 5: stats[a]['last_5'].pop(0)

    features_df = pd.DataFrame(features_list)
    df_final = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    
    return df_final, stats

def rodar_robo():
    # 1. COLETA MASSIVA DE DADOS (Passado + Presente)
    print("\n2. Coletando dados para treinamento...")
    
    # Busca temporada passada completa
    df_historico = coletar_jogos_realizados(SEASON_TRAIN_HISTORIC)
    
    # Busca jogos JÁ REALIZADOS da temporada atual
    df_atual = coletar_jogos_realizados(SEASON_CURRENT)
    
    # Junta tudo num único dataset cronológico
    if df_historico.empty and df_atual.empty:
        print("❌ Erro Crítico: Nenhum dado encontrado para treinar.")
        return

    df_treino_total = pd.concat([df_historico, df_atual], ignore_index=True)
    df_treino_total = df_treino_total.sort_values('date')
    
    print(f"   -> Total de jogos para aprendizado: {len(df_treino_total)}")

    # 2. ENGENHARIA E TREINAMENTO
    # O 'current_stats_today' conterá o estado dos times APÓS o último jogo da lista (HOJE)
    df_enriched, current_stats_today = engenharia_de_features(df_treino_total)
    
    FEATURE_COLS = ['points_diff', 'home_form', 'away_form', 'home_attack', 'away_defense', 'away_attack', 'home_defense']
    X = df_enriched[FEATURE_COLS]
    y = df_enriched['result']
    
    print("3. Treinando Modelo com dados combinados...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # 3. PREVISÃO DO FUTURO
    print(f"\n4. Buscando próximos jogos ({SEASON_CURRENT})...")
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {'x-apisports-key': API_KEY}
    # Busca os próximos 10 que ainda NÃO começaram (NS = Not Started)
    params = {"league": LEAGUE_ID, "season": SEASON_CURRENT, "next": "10"}
    
    try:
        resp = requests.get(url, headers=headers, params=params)
        data_next = resp.json()
    except:
        data_next = {'results': 0}
    
    jogos_futuros = []
    
    if data_next.get('results', 0) > 0:
        jogos_futuros = data_next['response']
        print(f"✅ {len(jogos_futuros)} jogos futuros encontrados.")
    else:
        print("⚠️ Nenhum jogo agendado na API. Entrando em modo SIMULAÇÃO.")
        # Simula com os últimos 5 jogos reais apenas para não quebrar o App
        last_games = df_treino_total.tail(5).to_dict('records')
        for lg in last_games:
            jogos_futuros.append({
                'fixture': {'id': f"sim_{lg['date']}", 'date': str(lg['date']), 'venue': {'name': 'Simulação (Sem Jogos Reais)'}, 'status': {'short': 'NS'}},
                'teams': {'home': {'name': lg['home_team']}, 'away': {'name': lg['away_team']}},
                'is_simulation': True
            })

    if not firebase_admin._apps:
        print("❌ Sem conexão Firebase. Fim.")
        return

    batch = db.batch()
    print("5. Calculando e Salvando...")
    
    def safe_get_stat(team, metric, default=0):
        # Busca no dicionário 'current_stats_today' que reflete o HOJE
        if team not in current_stats_today: return default
        s = current_stats_today[team]
        if metric == 'form': return sum(s['last_5'])
        if s['games'] == 0: return 0
        return s[metric] / s['games']

    for item in jogos_futuros:
        home = item['teams']['home']['name']
        away = item['teams']['away']['name']
        
        # Prepara as features usando as estatísticas ATUAIS
        h_form = safe_get_stat(home, 'form')
        a_form = safe_get_stat(away, 'form')
        h_att = safe_get_stat(home, 'goals_scored')
        h_def = safe_get_stat(home, 'goals_conceded')
        a_att = safe_get_stat(away, 'goals_scored')
        a_def = safe_get_stat(away, 'goals_conceded')
        
        h_pts = current_stats_today.get(home, {}).get('points', 0)
        a_pts = current_stats_today.get(away, {}).get('points', 0)
        
        features_jogo = [[
            h_pts - a_pts,
            h_form, a_form,
            h_att, a_def,
            a_att, h_def
        ]]
        
        probs = model.predict_proba(features_jogo)[0]
        
        # Formatação para o App
        def points_to_str(pts):
            mapa = {3:'V', 1:'E', 0:'D'}
            return "".join([mapa.get(p, '-') for p in pts])
        
        insight = "Confronto equilibrado."
        if probs[1] > 0.6: insight = f"{home} é favorito! Ataque letal de {h_att:.1f} gols/jogo."
        elif probs[2] > 0.6: insight = f"{away} deve vencer, aproveitando a defesa frágil do rival."
        elif abs(h_form - a_form) > 4: insight = f"O momento favorece o {home if h_form > a_form else away}."
        
        game_id = str(item['fixture']['id'])
        doc_ref = db.collection('games').document(game_id)
        
        ts = int(datetime.datetime.now().timestamp() * 1000)
        try:
            dt = pd.to_datetime(item['fixture']['date'])
            date_fmt = dt.strftime("%d/%m %H:%M")
            ts = int(dt.timestamp() * 1000)
        except:
            date_fmt = "Data a confirmar"

        dados = {
            'id': game_id,
            'homeTeam': home, 'awayTeam': away,
            'date': date_fmt,
            'venue': item['fixture']['venue']['name'] or "Estádio",
            'homeForm': points_to_str(current_stats_today.get(home, {}).get('last_5', [])),
            'awayForm': points_to_str(current_stats_today.get(away, {}).get('last_5', [])),
            'probs': {
                'home': int(probs[1] * 100),
                'draw': int(probs[0] * 100),
                'away': int(probs[2] * 100)
            },
            # --- ESTATÍSTICAS PARA O APP ---
            'stats': {
                'homeAttack': float(f"{h_att:.2f}"),
                'homeDefense': float(f"{h_def:.2f}"),
                'awayAttack': float(f"{a_att:.2f}"),
                'awayDefense': float(f"{a_def:.2f}")
            },
            'insight': insight,
            'timestamp': ts
        }
        batch.set(doc_ref, dados)

    batch.commit()
    print("✅ CICLO COMPLETO: Banco de dados atualizado!")

if __name__ == "__main__":
    rodar_robo()
