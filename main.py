import os
import json
import requests
import pandas as pd
import numpy as np
import datetime
import re
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------------------------
# 1. CONFIGURAÇÕES (API-FOOTBALL)
# -------------------------------------------------------------------------
# Coloque a sua NOVA chave aqui
API_KEY = os.environ.get("FOOTBALL_API_KEY", "SUA_NOVA_API_KEY_AQUI") 

# IDs da API-Football (api-sports.io)
LEAGUES = {
    '71': 'Brasileirão Série A',
    '39': 'Premier League (ING)',
    '140': 'La Liga (ESP)',
    '2': 'Champions League',
    '13': 'Copa Libertadores',
    '11': 'Copa Sul-Americana'
}

# Ano para análise (ajuste conforme a disponibilidade da sua conta free)
# Dica: Se a conta é nova, tente 2023 ou 2024.
SEASON_TARGET = "2023" 

print(f"⚙️ ROBÔ API-FOOTBALL INICIADO (Season {SEASON_TARGET})")

# -------------------------------------------------------------------------
# CONEXÃO FIREBASE
# -------------------------------------------------------------------------
print("1. Conectando ao Firebase...", flush=True)
if not firebase_admin._apps:
    firebase_creds_str = os.environ.get("FIREBASE_CREDENTIALS")
    if firebase_creds_str:
        cred = credentials.Certificate(json.loads(firebase_creds_str))
        firebase_admin.initialize_app(cred)
    else:
        local_key_path = "firebase_key.json" 
        if os.path.exists(local_key_path):
            cred = credentials.Certificate(local_key_path)
            firebase_admin.initialize_app(cred)
        else:
            print("⚠️ AVISO: Sem credenciais. O script não salvará nada.", flush=True)

if firebase_admin._apps:
    db = firestore.client()
    print("✅ Conectado ao banco de dados!", flush=True)

# -------------------------------------------------------------------------
# LÓGICA DE DADOS
# -------------------------------------------------------------------------

def coletar_campeonato(league_id, league_name):
    print(f"   -> Baixando {league_name} (ID {league_id})...", flush=True)
    
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {'x-apisports-key': API_KEY}
    params = {"league": league_id, "season": SEASON_TARGET} 
    
    try:
        resp = requests.get(url, headers=headers, params=params)
        data = resp.json()
        
        if "errors" in data and data["errors"]:
            print(f"      ❌ Erro API: {data['errors']}", flush=True)
            return pd.DataFrame()
            
        jogos = []
        if 'response' in data:
            for item in data['response']:
                # Tratamento de Rodada
                rodada_str = item['league']['round']
                # Extrai apenas os números da string "Regular Season - 12"
                numeros = re.findall(r'\d+', str(rodada_str))
                rodada_num = int(numeros[0]) if numeros else 0
                
                # Tratamento especial para Copas (Finais, Semis)
                if "Final" in rodada_str: rodada_num = 50
                if "Semi" in rodada_str: rodada_num = 49
                if "Quarter" in rodada_str: rodada_num = 48
                if "16" in rodada_str or "8th" in rodada_str: rodada_num = 47

                home = item['goals']['home']
                away = item['goals']['away']
                status_short = item['fixture']['status']['short']
                
                # Logos
                home_logo = item['teams']['home']['logo']
                away_logo = item['teams']['away']['logo']

                result = None
                # Considera jogo terminado se tiver status FT/AET/PEN
                if status_short in ['FT', 'AET', 'PEN'] and home is not None:
                    result = 1 if home > away else (2 if away > home else 0)
                
                jogos.append({
                    'id': str(item['fixture']['id']),
                    'league_id': league_id,
                    'date': item['fixture']['date'], # Formato ISO
                    'home_team': item['teams']['home']['name'],
                    'away_team': item['teams']['away']['name'],
                    'home_logo': home_logo,
                    'away_logo': away_logo,
                    'home_goals': home,
                    'away_goals': away,
                    'result': result,
                    'venue': item['fixture']['venue']['name'],
                    'round': rodada_num,
                    'round_label': rodada_str,
                    'status': status_short
                })
        
        df = pd.DataFrame(jogos)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            print(f"      ✅ Sucesso: {len(df)} jogos encontrados.", flush=True)
        return df

    except Exception as e:
        print(f"      ❌ Erro de Conexão: {e}", flush=True)
        return pd.DataFrame()

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
            if stats[team]['games'] == 0: return 0.0
            return float(stats[team][metric] / stats[team]['games'])

        def get_form(team):
            return sum(stats[team]['last_5'])

        features = {
            'home_attack': get_avg(h, 'goals_scored'),
            'away_defense': get_avg(a, 'goals_conceded'),
            'away_attack': get_avg(a, 'goals_scored'),
            'home_defense': get_avg(h, 'goals_conceded'),
            'home_form_val': get_form(h),
            'away_form_val': get_form(a),
            'diff_points': stats[h]['points'] - stats[a]['points']
        }
        features_list.append(features)

        # Atualiza Stats se o jogo terminou
        if row['result'] is not None:
            gh = int(row['home_goals'])
            ga = int(row['away_goals'])
            
            stats[h]['games'] += 1; stats[a]['games'] += 1
            stats[h]['goals_scored'] += gh; stats[h]['goals_conceded'] += ga
            stats[a]['goals_scored'] += ga; stats[a]['goals_conceded'] += gh
            
            # Recalcula pontos para garantir precisão
            if gh > ga: pts_h, pts_a = 3, 0
            elif ga > gh: pts_h, pts_a = 0, 3
            else: pts_h, pts_a = 1, 1
            
            stats[h]['points'] += pts_h; stats[a]['points'] += pts_a
            stats[h]['last_5'].append(pts_h)
            if len(stats[h]['last_5']) > 5: stats[h]['last_5'].pop(0)
            stats[a]['last_5'].append(pts_a)
            if len(stats[a]['last_5']) > 5: stats[a]['last_5'].pop(0)

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(features_list)], axis=1)

def rodar_robo():
    if not firebase_admin._apps: return
    batch = db.batch()
    count_total = 0
    count_batch = 0
    
    for league_id, league_name in LEAGUES.items():
        print(f"\n--- Processando: {league_name} ---", flush=True)
        
        df = coletar_campeonato(league_id, league_name)
        if df.empty: continue
        
        df_enriched = engenharia_de_features(df)
        
        # Treina IA
        df_treino = df_enriched.dropna(subset=['result'])
        model = None
        if len(df_treino) > 10:
            X = df_treino[['diff_points', 'home_form_val', 'away_form_val', 'home_attack', 'home_defense', 'away_attack', 'away_defense']]
            y = df_treino['result'].astype(int)
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X, y)

        print(f"   Salvando {len(df_enriched)} jogos no Firebase...", flush=True)
        
        for index, row in df_enriched.iterrows():
            probs = {'home': 33, 'draw': 34, 'away': 33}
            insight = "Aguardando dados."
            
            if model:
                try:
                    feats = [[
                        row['diff_points'], row['home_form_val'], row['away_form_val'],
                        row['home_attack'], row['home_defense'], row['away_attack'], row['away_defense']
                    ]]
                    if not np.isnan(feats).any():
                        p = model.predict_proba(feats)[0]
                        probs = {'home': int(p[1]*100), 'draw': int(p[0]*100), 'away': int(p[2]*100)}
                        
                        if p[1] > 0.60: insight = f"{row['home_team']} tem grande favoritismo em casa."
                        elif p[2] > 0.60: insight = f"{row['away_team']} deve vencer, mesmo fora."
                        else: insight = "Confronto muito equilibrado."
                except:
                    pass

            ts = int(row['date'].timestamp() * 1000)
            date_fmt = row['date'].strftime("%d/%m %H:%M")
            
            doc_ref = db.collection('games').document(row['id'])
            
            status_final = row['status']
            score_h = int(row['home_goals']) if pd.notna(row['home_goals']) else None
            score_a = int(row['away_goals']) if pd.notna(row['away_goals']) else None

            dados = {
                'id': row['id'],
                'leagueId': league_id,
                'leagueName': league_name,
                'round': int(row['round']) if pd.notna(row['round']) else 0,
                'roundLabel': str(row['round_label']),
                'homeTeam': str(row['home_team']), 'awayTeam': str(row['away_team']),
                'homeLogo': str(row['home_logo']), 'awayLogo': str(row['away_logo']),
                'homeScore': score_h,
                'awayScore': score_a,
                'date': date_fmt,
                'venue': str(row['venue']),
                'probs': probs,
                'stats': {
                    'homeAttack': float(f"{row['home_attack']:.2f}"), 
                    'homeDefense': float(f"{row['home_defense']:.2f}"), 
                    'awayAttack': float(f"{row['away_attack']:.2f}"), 
                    'awayDefense': float(f"{row['away_defense']:.2f}"),
                    'isMock': False
                },
                'insight': insight,
                'timestamp': ts,
                'status': status_final
            }
            batch.set(doc_ref, dados)
            count_total += 1
            count_batch += 1
            
            if count_batch >= 400:
                batch.commit()
                batch = db.batch()
                count_batch = 0
                print(f"   ... comitando lote...", flush=True)

    if count_batch > 0:
        batch.commit()

    print(f"\n✅ SUCESSO! Total de jogos salvos: {count_total}", flush=True)

if __name__ == "__main__":
    rodar_robo()
