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
# 1. CONFIGURAÇÕES
# -------------------------------------------------------------------------
API_KEY = os.environ.get("FOOTBALL_API_KEY", "SUA_API_KEY_AQUI") 
LEAGUE_ID = "71"  # Brasileirão Série A

hoje = datetime.datetime.now()
ANO_ATUAL = hoje.year
SEASON_CURRENT = str(ANO_ATUAL)
SEASON_TRAIN_HISTORIC = str(ANO_ATUAL - 1)

print(f"⚙️ Robô Iniciado: {hoje.strftime('%d/%m/%Y')}")

# -------------------------------------------------------------------------
# 2. CONEXÃO FIREBASE
# -------------------------------------------------------------------------
print("1. Conectando ao Firebase...")
if not firebase_admin._apps:
    firebase_creds_str = os.environ.get("FIREBASE_CREDENTIALS")
    if firebase_creds_str:
        cred = credentials.Certificate(json.loads(firebase_creds_str))
        firebase_admin.initialize_app(cred)
    else:
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
# 3. LÓGICA DE IA
# -------------------------------------------------------------------------

def coletar_jogos_realizados(season):
    print(f"   -> Baixando histórico de {season}...")
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {'x-apisports-key': API_KEY}
    params = {"league": LEAGUE_ID, "season": season, "status": "FT"} 
    
    try:
        resp = requests.get(url, headers=headers, params=params)
        data = resp.json()
    except:
        return pd.DataFrame()
    
    jogos = []
    if 'response' in data:
        for item in data['response']:
            home = item['goals']['home']
            away = item['goals']['away']
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
    stats = {}
    all_teams = set(df['home_team']).union(set(df['away_team']))
    
    for team in all_teams:
        stats[team] = {'points': 0, 'games': 0, 'goals_scored': 0, 'goals_conceded': 0, 'last_5': []}

    features_list = []

    for index, row in df.iterrows():
        h = row['home_team']
        a = row['away_team']
        
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

        stats[h]['games'] += 1; stats[a]['games'] += 1
        stats[h]['goals_scored'] += row['home_goals']; stats[h]['goals_conceded'] += row['away_goals']
        stats[a]['goals_scored'] += row['away_goals']; stats[a]['goals_conceded'] += row['home_goals']
        
        pts_h = 3 if row['result'] == 1 else (1 if row['result'] == 0 else 0)
        pts_a = 3 if row['result'] == 2 else (1 if row['result'] == 0 else 0)
            
        stats[h]['points'] += pts_h; stats[a]['points'] += pts_a
        
        stats[h]['last_5'].append(pts_h)
        if len(stats[h]['last_5']) > 5: stats[h]['last_5'].pop(0)
        stats[a]['last_5'].append(pts_a)
        if len(stats[a]['last_5']) > 5: stats[a]['last_5'].pop(0)

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(features_list)], axis=1), stats

def rodar_robo():
    print("\n2. Treinando Robô...")
    df_historico = coletar_jogos_realizados(SEASON_TRAIN_HISTORIC)
    df_atual = coletar_jogos_realizados(SEASON_CURRENT)
    
    if df_historico.empty and df_atual.empty:
        print("❌ Sem dados para treinar.")
        return

    df_treino = pd.concat([df_historico, df_atual], ignore_index=True).sort_values('date')
    df_enriched, current_stats = engenharia_de_features(df_treino)
    
    X = df_enriched[['points_diff', 'home_form', 'away_form', 'home_attack', 'away_defense', 'away_attack', 'home_defense']]
    y = df_enriched['result']
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # --- MUDANÇA CRÍTICA AQUI: BUSCA POR DATAS ---
    print(f"\n4. Buscando jogos de HOJE até +7 dias ({SEASON_CURRENT})...")
    
    data_inicio = datetime.date.today().strftime("%Y-%m-%d") # Hoje
    data_fim = (datetime.date.today() + datetime.timedelta(days=7)).strftime("%Y-%m-%d") # Daqui a 7 dias
    
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {'x-apisports-key': API_KEY}
    # "from" e "to" pega TUDO nesse intervalo
    params = {"league": LEAGUE_ID, "season": SEASON_CURRENT, "from": data_inicio, "to": data_fim}
    
    try:
        resp = requests.get(url, headers=headers, params=params)
        data_next = resp.json()
    except:
        data_next = {'results': 0}
    
    jogos_futuros = []
    if data_next.get('results', 0) > 0:
        jogos_futuros = data_next['response']
        print(f"✅ {len(jogos_futuros)} jogos encontrados na grade da semana.")
    else:
        print("⚠️ Nenhum jogo encontrado nesta semana. Usando simulação.")
        last_games = df_treino.tail(5).to_dict('records')
        for lg in last_games:
            jogos_futuros.append({
                'fixture': {'id': f"sim_{lg['date']}", 'date': str(lg['date']), 'venue': {'name': 'Simulação'}, 'status': {'short': 'NS'}},
                'teams': {'home': {'name': lg['home_team']}, 'away': {'name': lg['away_team']}},
                'is_simulation': True
            })

    if not firebase_admin._apps: return

    batch = db.batch()
    print("5. Salvando...")
    
    def safe_get_stat(team, metric):
        if team not in current_stats: return 0
        s = current_stats[team]
        if metric == 'form': return sum(s['last_5'])
        if s['games'] == 0: return 0
        return s[metric] / s['games']

    for item in jogos_futuros:
        # Pega apenas jogos Não Iniciados (NS) ou A Definir (TBD)
        # Se quiser incluir jogos AO VIVO, adicione '1H', '2H', 'HT' na lista abaixo
        status = item['fixture']['status']['short']
        if not item.get('is_simulation') and status not in ['NS', 'TBD', 'PST']: 
            continue 

        home = item['teams']['home']['name']
        away = item['teams']['away']['name']
        
        h_form = safe_get_stat(home, 'form')
        a_form = safe_get_stat(away, 'form')
        h_att = safe_get_stat(home, 'goals_scored')
        h_def = safe_get_stat(home, 'goals_conceded')
        a_att = safe_get_stat(away, 'goals_scored')
        a_def = safe_get_stat(away, 'goals_conceded')
        
        h_pts = current_stats.get(home, {}).get('points', 0)
        a_pts = current_stats.get(away, {}).get('points', 0)
        
        features_jogo = [[h_pts - a_pts, h_form, a_form, h_att, h_def, a_att, a_def]]
        probs = model.predict_proba(features_jogo)[0]
        
        def points_to_str(pts):
            mapa = {3:'V', 1:'E', 0:'D'}
            return "".join([mapa.get(p, '-') for p in pts])
        
        insight = "Jogo duro."
        if probs[1] > 0.6: insight = f"{home} favorito com ataque de {h_att:.1f} gols/jogo."
        elif probs[2] > 0.6: insight = f"{away} perigoso contra defesa do {home}."
        elif abs(h_form - a_form) > 4: insight = f"{home if h_form > a_form else away} vem em melhor fase."
        
        game_id = str(item['fixture']['id'])
        doc_ref = db.collection('games').document(game_id)
        
        ts = int(datetime.datetime.now().timestamp() * 1000)
        try:
            dt = pd.to_datetime(item['fixture']['date'])
            date_fmt = dt.strftime("%d/%m %H:%M")
            ts = int(dt.timestamp() * 1000)
        except:
            date_fmt = "Data a definir"

        dados = {
            'id': game_id,
            'homeTeam': home, 'awayTeam': away,
            'date': date_fmt,
            'venue': item['fixture']['venue']['name'] or "Estádio",
            'homeForm': points_to_str(current_stats.get(home, {}).get('last_5', [])),
            'awayForm': points_to_str(current_stats.get(away, {}).get('last_5', [])),
            'probs': {'home': int(probs[1]*100), 'draw': int(probs[0]*100), 'away': int(probs[2]*100)},
            'stats': {'homeAttack': float(f"{h_att:.2f}"), 'homeDefense': float(f"{h_def:.2f}"), 'awayAttack': float(f"{a_att:.2f}"), 'awayDefense': float(f"{a_def:.2f}")},
            'insight': insight,
            'timestamp': ts
        }
        batch.set(doc_ref, dados)

    batch.commit()
    print("✅ GRADE DA SEMANA ATUALIZADA!")

if __name__ == "__main__":
    rodar_robo()