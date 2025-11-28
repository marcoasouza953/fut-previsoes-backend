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
# CONFIGURAÇÕES
# -------------------------------------------------------------------------
API_KEY = os.environ.get("FOOTBALL_API_KEY", "SUA_API_KEY_AQUI") 
LEAGUE_ID = "71"  # Brasileirão Série A
SEASON_TRAIN = "2023"
SEASON_CURRENT = "2024"

# -------------------------------------------------------------------------
# CONEXÃO COM O FIREBASE
# -------------------------------------------------------------------------
print("1. Conectando ao Firebase...")
if not firebase_admin._apps:
    firebase_creds_str = os.environ.get("FIREBASE_CREDENTIALS")
    if firebase_creds_str:
        cred = credentials.Certificate(json.loads(firebase_creds_str))
    else:
        # Fallback local
        local_key_path = "firebase_key.json" 
        if os.path.exists(local_key_path):
            cred = credentials.Certificate(local_key_path)
        else:
            # Cria um app dummy para evitar erro se não tiver chave (apenas para teste de sintaxe)
            print("⚠️ AVISO: Sem credenciais. O código vai falhar ao tentar salvar.")
            pass # Em produção, isso deve falhar.

    if firebase_admin._apps or 'cred' in locals():
         if not firebase_admin._apps: firebase_admin.initialize_app(cred)
         db = firestore.client()
         print("✅ Conectado ao banco de dados!")

# -------------------------------------------------------------------------
# LÓGICA AVANÇADA DE IA
# -------------------------------------------------------------------------

def coletar_dados_da_temporada(season):
    print(f"   -> Baixando temporada {season}...")
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {'x-apisports-key': API_KEY}
    params = {"league": LEAGUE_ID, "season": season}
    
    resp = requests.get(url, headers=headers, params=params)
    data = resp.json()
    
    jogos = []
    if 'response' in data:
        for item in data['response']:
            # Pega jogos finalizados (FT)
            if item['fixture']['status']['short'] == 'FT':
                home = item['goals']['home']
                away = item['goals']['away']
                result = 1 if home > away else (2 if away > home else 0)
                
                jogos.append({
                    'date': item['fixture']['date'],
                    'home_team': item['teams']['home']['name'],
                    'away_team': item['teams']['away']['name'],
                    'home_goals': home,
                    'away_goals': away,
                    'result': result,
                    'is_future': False
                })
    
    df = pd.DataFrame(jogos)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    return df

def engenharia_de_features(df):
    """
    O SEGREDO DA IA: Aqui transformamos placares simples em estatísticas poderosas.
    Calculamos tudo jogo a jogo para simular o conhecimento que tínhamos NAQUELA data.
    """
    stats = {} # { 'Palmeiras': {'points': 45, 'games': 20, 'goals_scored': 30, 'goals_conceded': 15, 'last_5': [3, 1, 0, 3, 3]} }
    
    # Inicializa stats para todos os times
    all_teams = set(df['home_team']).union(set(df['away_team']))
    for team in all_teams:
        stats[team] = {
            'points': 0, 'games': 0, 
            'goals_scored': 0, 'goals_conceded': 0,
            'last_5': [] # Lista para guardar pontos dos últimos 5 jogos (3, 1 ou 0)
        }

    # Listas para guardar as features calculadas
    features_list = []

    for index, row in df.iterrows():
        h = row['home_team']
        a = row['away_team']
        
        # 1. EXTRAI AS FEATURES (O estado dos times ANTES do jogo começar)
        
        # Função auxiliar para calcular média segura (evitar divisão por zero)
        def get_avg(team, metric):
            if stats[team]['games'] == 0: return 0
            return stats[team][metric] / stats[team]['games']

        # Função auxiliar para calcular forma (soma dos últimos 5 jogos)
        def get_form(team):
            return sum(stats[team]['last_5'])

        # Monta a linha de dados para a IA
        features = {
            'home_points': stats[h]['points'],
            'away_points': stats[a]['points'],
            'points_diff': stats[h]['points'] - stats[a]['points'],
            
            # --- NOVAS FEATURES AVANÇADAS ---
            'home_form': get_form(h),          # Quão quente está o mandante?
            'away_form': get_form(a),          # Quão quente está o visitante?
            'form_diff': get_form(h) - get_form(a),
            
            'home_attack': get_avg(h, 'goals_scored'),      # Poder de fogo média
            'away_defense': get_avg(a, 'goals_conceded'),   # Fragilidade defensiva média
            'away_attack': get_avg(a, 'goals_scored'),
            'home_defense': get_avg(h, 'goals_conceded'),
        }
        features_list.append(features)

        # 2. ATUALIZA AS ESTATÍSTICAS (Com o resultado real deste jogo, para o próximo loop)
        
        # Atualiza contadores básicos
        stats[h]['games'] += 1
        stats[a]['games'] += 1
        stats[h]['goals_scored'] += row['home_goals']
        stats[h]['goals_conceded'] += row['away_goals']
        stats[a]['goals_scored'] += row['away_goals']
        stats[a]['goals_conceded'] += row['home_goals']
        
        # Pontos da partida
        pts_h, pts_a = 0, 0
        if row['result'] == 1: pts_h = 3
        elif row['result'] == 2: pts_a = 3
        else:
            pts_h, pts_a = 1, 1
            
        stats[h]['points'] += pts_h
        stats[a]['points'] += pts_a
        
        # Atualiza a fila dos últimos 5 jogos (Remove o mais antigo se já tiver 5)
        stats[h]['last_5'].append(pts_h)
        if len(stats[h]['last_5']) > 5: stats[h]['last_5'].pop(0)
        
        stats[a]['last_5'].append(pts_a)
        if len(stats[a]['last_5']) > 5: stats[a]['last_5'].pop(0)

    # Anexa as features calculadas ao DataFrame original
    features_df = pd.DataFrame(features_list)
    df_final = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    
    return df_final, stats

def treinar_e_prever():
    # 1. PREPARAÇÃO (TREINO)
    print("\n2. Preparando IA com dados de 2023...")
    df_train = coletar_dados_da_temporada(SEASON_TRAIN)
    if df_train.empty:
        print("❌ Erro: Não foi possível baixar dados de treino.")
        return

    # Aplica a engenharia de features nos dados de treino
    df_train_enriched, _ = engenharia_de_features(df_train)
    
    # Define quais colunas a IA vai olhar
    FEATURE_COLS = ['points_diff', 'home_form', 'away_form', 'home_attack', 'away_defense', 'away_attack', 'home_defense']
    
    X = df_train_enriched[FEATURE_COLS]
    y = df_train_enriched['result']
    
    # Treina o modelo (Aumentei n_estimators para 100 para mais precisão)
    print("3. Treinando Random Forest com estatísticas avançadas...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # 2. PREPARAÇÃO (JOGOS REAIS/ATUAIS)
    print(f"\n4. Buscando contexto da temporada atual ({SEASON_CURRENT})...")
    df_current = coletar_dados_da_temporada(SEASON_CURRENT)
    
    # Precisamos rodar a engenharia na temporada atual inteira para ter o estado ATUAL dos times (stats)
    # Mesmo que o df_current esteja vazio (início de temporada), o código trata.
    if not df_current.empty:
        _, current_stats = engenharia_de_features(df_current)
    else:
        # Se não tiver jogos em 2024, usamos simulação ou stats zerados
        current_stats = {t: {'points':0, 'games':0, 'goals_scored':0, 'goals_conceded':0, 'last_5':[]} for t in set(df_train['home_team'])}

    # 3. BUSCA JOGOS FUTUROS
    print("5. Buscando próximos jogos agendados...")
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {'x-apisports-key': API_KEY}
    params = {"league": LEAGUE_ID, "season": SEASON_CURRENT, "next": "10"}
    
    resp = requests.get(url, headers=headers, params=params)
    data_next = resp.json()
    
    jogos_futuros = []
    
    if data_next['results'] > 0:
        jogos_futuros = data_next['response']
        print(f"✅ {len(jogos_futuros)} jogos encontrados.")
    else:
        print("⚠️ Sem jogos agendados. Usando SIMULAÇÃO (últimos do treino).")
        # Simula com os últimos 5 do treino para não deixar o app vazio
        last_games = df_train.tail(5).to_dict('records')
        for lg in last_games:
            # Estrutura fake para simular a resposta da API
            jogos_futuros.append({
                'fixture': {'id': f"sim_{lg['date']}", 'date': str(lg['date']), 'venue': {'name': 'Simulação'}, 'status': {'short': 'NS'}},
                'teams': {'home': {'name': lg['home_team']}, 'away': {'name': lg['away_team']}},
                'is_simulation': True
            })

    # 4. PREVISÃO E SALVAMENTO
    batch = db.batch()
    print("6. Calculando probabilidades e salvando...")
    
    def safe_get_stat(team, metric, default=0):
        # Auxiliar para pegar stat do time atual, ou 0 se time novo
        if team not in current_stats: return default
        s = current_stats[team]
        if metric == 'form': return sum(s['last_5'])
        if s['games'] == 0: return 0
        return s[metric] / s['games']

    for item in jogos_futuros:
        home = item['teams']['home']['name']
        away = item['teams']['away']['name']
        
        # Constrói as features para o jogo futuro usando o current_stats
        h_form = safe_get_stat(home, 'form')
        a_form = safe_get_stat(away, 'form')
        h_att = safe_get_stat(home, 'goals_scored')
        a_def = safe_get_stat(away, 'goals_conceded')
        a_att = safe_get_stat(away, 'goals_scored')
        h_def = safe_get_stat(home, 'goals_conceded')
        
        # Pontos totais para diff
        h_pts = current_stats.get(home, {}).get('points', 0)
        a_pts = current_stats.get(away, {}).get('points', 0)
        
        # O vetor X deve ter a mesma ordem do treino:
        # ['points_diff', 'home_form', 'away_form', 'home_attack', 'away_defense', 'away_attack', 'home_defense']
        features_jogo = [[
            h_pts - a_pts,
            h_form, a_form,
            h_att, a_def,
            a_att, h_def
        ]]
        
        # Previsão
        probs = model.predict_proba(features_jogo)[0]
        
        # Formata forma para exibir no App (ex: "VVDDE")
        # Como temos apenas os pontos (3,1,0), vamos converter toscamente para string
        def points_to_str(pts_list):
            mapa = {3:'V', 1:'E', 0:'D'}
            return "".join([mapa.get(p, '-') for p in pts_list])
        
        h_form_str = points_to_str(current_stats.get(home, {}).get('last_5', []))
        a_form_str = points_to_str(current_stats.get(away, {}).get('last_5', []))
        
        # Cria insight textual
        insight = "Jogo equilibrado."
        if probs[1] > 0.6: insight = f"{home} é favorito com ataque forte ({h_att:.1f} gols/jogo)."
        elif probs[2] > 0.6: insight = f"{away} tem grande chance, aproveitando má defesa do rival."
        elif abs(h_form - a_form) > 5: insight = f"Momento decisivo: {home if h_form > a_form else away} vem em fase muito melhor."
        
        # Salva
        game_id = str(item['fixture']['id'])
        doc_ref = db.collection('games').document(game_id)
        
        # Data
        date_str = item['fixture']['date']
        ts = 0
        try:
            dt = pd.to_datetime(date_str)
            date_fmt = dt.strftime("%d/%m %H:%M")
            ts = int(dt.timestamp() * 1000)
        except:
            date_fmt = "Data a confirmar"

        dados = {
            'id': game_id,
            'homeTeam': home, 'awayTeam': away,
            'date': date_fmt,
            'venue': item['fixture']['venue']['name'] or "Estádio",
            'homeForm': h_form_str or "-----",
            'awayForm': a_form_str or "-----",
            'probs': {
                'home': int(probs[1] * 100),
                'draw': int(probs[0] * 100),
                'away': int(probs[2] * 100)
            },
            'insight': insight,
            'timestamp': ts
        }
        batch.set(doc_ref, dados)

    batch.commit()
    print("✅ Previsões AVANÇADAS enviadas com sucesso!")

if __name__ == "__main__":
    if 'db' in globals() or firebase_admin._apps:
        treinar_e_prever()
    else:
        print("❌ Erro de conexão com Firebase. Verifique as chaves.")
