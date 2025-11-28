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
# Tenta pegar a chave do ambiente (GitHub Actions), senão usa a string direta
API_KEY = os.environ.get("FOOTBALL_API_KEY", "SUA_API_KEY_AQUI") 

LEAGUE_ID = "71"  # Brasileirão Série A
SEASON_TRAIN = "2023" # Ano para treinar a IA
SEASON_CURRENT = "2024" # Ano atual para buscar jogos reais

# -------------------------------------------------------------------------
# CONEXÃO COM O FIREBASE
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
            print("⚠️ AVISO: Sem credenciais encontradas. O salvamento vai falhar.")

if firebase_admin._apps:
    db = firestore.client()
    print("✅ Conectado ao banco de dados!")

# -------------------------------------------------------------------------
# LÓGICA DE INTELIGÊNCIA ARTIFICIAL
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
            # Apenas jogos terminados contam para estatística
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
                    'result': result
                })
    
    df = pd.DataFrame(jogos)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    return df

def engenharia_de_features(df):
    """
    Calcula estatísticas avançadas (Ataque, Defesa, Forma) jogo a jogo.
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

    for index, row in df.iterrows():
        h = row['home_team']
        a = row['away_team']
        
        # Funções auxiliares para calcular médias no momento do jogo
        def get_avg(team, metric):
            if stats[team]['games'] == 0: return 0
            return stats[team][metric] / stats[team]['games']

        def get_form(team):
            return sum(stats[team]['last_5'])

        # Cria as 'features' (dados que a IA usa para aprender)
        features = {
            'home_points': stats[h]['points'],
            'away_points': stats[a]['points'],
            'points_diff': stats[h]['points'] - stats[a]['points'],
            
            # --- ESTATÍSTICAS PARA O APP ---
            'home_form': get_form(h),
            'away_form': get_form(a),
            'home_attack': get_avg(h, 'goals_scored'),      # Média de gols feitos
            'away_defense': get_avg(a, 'goals_conceded'),   # Média de gols sofridos
            'away_attack': get_avg(a, 'goals_scored'),
            'home_defense': get_avg(h, 'goals_conceded'),
        }
        features_list.append(features)

        # Atualiza os acumuladores com o resultado deste jogo
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
        
        # Mantém apenas os últimos 5 jogos na lista de forma
        stats[h]['last_5'].append(pts_h)
        if len(stats[h]['last_5']) > 5: stats[h]['last_5'].pop(0)
        stats[a]['last_5'].append(pts_a)
        if len(stats[a]['last_5']) > 5: stats[a]['last_5'].pop(0)

    # Junta tudo num DataFrame final
    features_df = pd.DataFrame(features_list)
    df_final = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    
    return df_final, stats

def rodar_robo():
    # 1. TREINAMENTO
    print("\n2. Preparando IA com dados de 2023...")
    df_train = coletar_dados_da_temporada(SEASON_TRAIN)
    if df_train.empty:
        print("❌ Erro: Sem dados de treino.")
        return

    # Enriquece os dados de treino com as estatísticas
    df_train_enriched, _ = engenharia_de_features(df_train)
    
    # Define o que a IA deve olhar
    FEATURE_COLS = ['points_diff', 'home_form', 'away_form', 'home_attack', 'away_defense', 'away_attack', 'home_defense']
    X = df_train_enriched[FEATURE_COLS]
    y = df_train_enriched['result']
    
    print("3. Treinando Modelo (Random Forest)...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # 2. CONTEXTO ATUAL
    print(f"\n4. Calculando força atual dos times ({SEASON_CURRENT})...")
    df_current = coletar_dados_da_temporada(SEASON_CURRENT)
    
    if not df_current.empty:
        # Se já começou a temporada 2024, calculamos as stats atuais
        _, current_stats = engenharia_de_features(df_current)
    else:
        # Se não, usamos stats vazias ou do final de 2023
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
        print(f"✅ {len(jogos_futuros)} jogos reais encontrados.")
    else:
        print("⚠️ Sem jogos agendados. Usando SIMULAÇÃO (últimos do treino).")
        last_games = df_train.tail(5).to_dict('records')
        for lg in last_games:
            jogos_futuros.append({
                'fixture': {'id': f"sim_{lg['date']}", 'date': str(lg['date']), 'venue': {'name': 'Simulação'}, 'status': {'short': 'NS'}},
                'teams': {'home': {'name': lg['home_team']}, 'away': {'name': lg['away_team']}},
                'is_simulation': True
            })

    # 4. PREVISÃO E SALVAMENTO
    if not firebase_admin._apps:
        print("❌ Sem conexão com Firebase. Pulando salvamento.")
        return

    batch = db.batch()
    print("6. Calculando probabilidades e salvando no Firebase...")
    
    # Função auxiliar para pegar stat de forma segura
    def safe_get_stat(team, metric, default=0):
        if team not in current_stats: return default
        s = current_stats[team]
        if metric == 'form': return sum(s['last_5'])
        if s['games'] == 0: return 0
        return s[metric] / s['games']

    for item in jogos_futuros:
        home = item['teams']['home']['name']
        away = item['teams']['away']['name']
        
        # Pega as stats atuais dos times
        h_form = safe_get_stat(home, 'form')
        a_form = safe_get_stat(away, 'form')
        h_att = safe_get_stat(home, 'goals_scored')
        h_def = safe_get_stat(home, 'goals_conceded')
        a_att = safe_get_stat(away, 'goals_scored')
        a_def = safe_get_stat(away, 'goals_conceded')
        
        h_pts = current_stats.get(home, {}).get('points', 0)
        a_pts = current_stats.get(away, {}).get('points', 0)
        
        # Monta a linha para a IA prever
        features_jogo = [[
            h_pts - a_pts,
            h_form, a_form,
            h_att, a_def,
            a_att, h_def
        ]]
        
        # A IA dá a probabilidade
        probs = model.predict_proba(features_jogo)[0]
        
        # Gera o Insight de texto
        insight = "Duelo equilibrado."
        if probs[1] > 0.6: insight = f"{home} favorito! Ataque potente ({h_att:.1f} gols/jogo)."
        elif probs[2] > 0.6: insight = f"{away} pode surpreender, defesa adversária frágil."
        elif h_form > a_form + 5: insight = f"{home} vem numa sequência muito melhor."
        
        # Prepara objeto para o App
        game_id = str(item['fixture']['id'])
        doc_ref = db.collection('games').document(game_id)
        
        # Tratamento de data
        ts = int(datetime.datetime.now().timestamp() * 1000)
        try:
            dt = pd.to_datetime(item['fixture']['date'])
            date_fmt = dt.strftime("%d/%m %H:%M")
            ts = int(dt.timestamp() * 1000)
        except:
            date_fmt = "Data a definir"

        # Converte forma para string "VVDDE"
        def points_to_str(pts):
            mapa = {3:'V', 1:'E', 0:'D'}
            return "".join([mapa.get(p, '-') for p in pts])

        dados = {
            'id': game_id,
            'homeTeam': home, 'awayTeam': away,
            'date': date_fmt,
            'venue': item['fixture']['venue']['name'] or "Estádio",
            'homeForm': points_to_str(current_stats.get(home, {}).get('last_5', [])),
            'awayForm': points_to_str(current_stats.get(away, {}).get('last_5', [])),
            'probs': {
                'home': int(probs[1] * 100),
                'draw': int(probs[0] * 100),
                'away': int(probs[2] * 100)
            },
            # --- ESTATÍSTICAS NOVAS (O que faltava) ---
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
    print("✅ SUCESSO! Dados atualizados no Firebase.")

if __name__ == "__main__":
    rodar_robo()
