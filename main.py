import os
import json
import sys
import traceback
import requests
import pandas as pd
import numpy as np
import datetime
import re
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------------------------
# 1. CONFIGURAÇÕES
# -------------------------------------------------------------------------
API_KEY = os.environ.get("FOOTBALL_API_KEY", "SUA_NOVA_API_KEY_AQUI") 

LEAGUES = {
    '71': 'Brasileirão Série A',
    '39': 'Premier League (ING)',
    '140': 'La Liga (ESP)',
    '2': 'Champions League',
    '13': 'Copa Libertadores',
    '11': 'Copa Sul-Americana'
}

# Forçar 2023 para garantir dados na conta Free
SEASON_TARGET = "2023" 

print(f"⚙️ ROBÔ PREVISÃO DE GOLOS: Baixando dados de {SEASON_TARGET}", flush=True)

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

def coletar_e_salvar_tabela(league_id, league_name):
    print(f"   -> Baixando Classificação de {league_name}...", flush=True)
    url = "https://v3.football.api-sports.io/standings"
    headers = {'x-apisports-key': API_KEY}
    params = {"league": league_id, "season": SEASON_TARGET}
    
    try:
        resp = requests.get(url, headers=headers, params=params)
        data = resp.json()
        
        if not data.get('response'):
            print("      ⚠️ Tabela não encontrada.", flush=True)
            return

        standings_data = []
        for league_data in data['response']:
            for group in league_data['league']['standings']:
                for team_rank in group:
                    standings_data.append({
                        'rank': team_rank['rank'],
                        'teamName': team_rank['team']['name'],
                        'teamLogo': team_rank['team']['logo'],
                        'points': team_rank['points'],
                        'goalsDiff': team_rank['goalsDiff'],
                        'played': team_rank['all']['played'],
                        'win': team_rank['all']['win'],
                        'draw': team_rank['all']['draw'],
                        'lose': team_rank['all']['lose'],
                        'form': team_rank.get('form', ''),
                        'group': league_data['league'].get('group', 'Único') or team_rank.get('group', 'Único')
                    })
        
        if firebase_admin._apps and standings_data:
            doc_ref = db.collection('standings').document(str(league_id))
            doc_ref.set({
                'leagueName': league_name,
                'updatedAt': int(datetime.datetime.now().timestamp() * 1000),
                'table': standings_data
            })
            print(f"      ✅ Tabela salva.", flush=True)

    except Exception as e:
        print(f"      ❌ Erro tabela: {e}", flush=True)

def coletar_campeonato(league_id, league_name):
    print(f"   -> Baixando Jogos de {league_name}...", flush=True)
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
                rodada_str = item['league']['round']
                numeros = re.findall(r'\d+', str(rodada_str))
                rodada_num = int(numeros[0]) if numeros else 0
                if "Final" in rodada_str: rodada_num = 50
                if "Semi" in rodada_str: rodada_num = 49
                if "Quarter" in rodada_str: rodada_num = 48
                if "16" in rodada_str: rodada_num = 47

                home = item['goals']['home']
                away = item['goals']['away']
                status = item['fixture']['status']['short']
                
                # Logos
                home_logo = item['teams']['home']['logo']
                away_logo = item['teams']['away']['logo']
                
                result = None
                # Para previsão de gols: 1 = Over 2.5, 0 = Under 2.5
                over_25 = None
                
                if status in ['FT', 'AET', 'PEN'] and home is not None:
                    result = 1 if home > away else (2 if away > home else 0)
                    total_goals = int(home) + int(away)
                    over_25 = 1 if total_goals > 2.5 else 0
                
                jogos.append({
                    'id': str(item['fixture']['id']),
                    'league_id': league_id,
                    'date': item['fixture']['date'],
                    'home_team': item['teams']['home']['name'],
                    'away_team': item['teams']['away']['name'],
                    'home_logo': home_logo,
                    'away_logo': away_logo,
                    'home_goals': home,
                    'away_goals': away,
                    'result': result,
                    'over_25': over_25, # NOVO ALVO
                    'venue': item['fixture']['venue']['name'] if item['fixture']['venue']['name'] else "Estádio",
                    'round': rodada_num,
                    'round_label': rodada_str,
                    'status': status
                })
        
        df = pd.DataFrame(jogos)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            print(f"      ✅ {len(df)} jogos encontrados.", flush=True)
        else:
            print(f"      ⚠️ Nenhum jogo encontrado para {SEASON_TARGET}.", flush=True)
        return df

    except Exception as e:
        print(f"      ❌ Erro de Conexão: {e}", flush=True)
        return pd.DataFrame()

def calcular_historico_h2h(df_completo, time_a, time_b, data_limite):
    mask = ((df_completo['home_team'] == time_a) & (df_completo['away_team'] == time_b)) | \
           ((df_completo['home_team'] == time_b) & (df_completo['away_team'] == time_a))
    past_games = df_completo[mask & (df_completo['date'] < data_limite) & (df_completo['status'] == 'FT')]
    ultimos = past_games.sort_values('date', ascending=False).head(3)
    
    historico = []
    for _, row in ultimos.iterrows():
        winner = 'Empate'
        if row['result'] == 1: winner = row['home_team']
        elif row['result'] == 2: winner = row['away_team']
        
        hg = int(row['home_goals']) if pd.notna(row['home_goals']) else 0
        ag = int(row['away_goals']) if pd.notna(row['away_goals']) else 0
        
        historico.append({
            'date': row['date'].strftime("%d/%m/%y"),
            'home': row['home_team'], 'away': row['away_team'],
            'score': f"{hg} - {ag}",
            'winner': winner
        })
    return historico

def engenharia_de_features(df):
    stats = {}
    all_teams = set(df['home_team']).union(set(df['away_team']))
    for team in all_teams: stats[team] = {'points': 0, 'games': 0, 'goals_scored': 0, 'goals_conceded': 0, 'last_5': []}
    features_list = []
    for index, row in df.iterrows():
        h, a = row['home_team'], row['away_team']
        def get_avg(t, m): return float(stats[t][m]/stats[t]['games']) if stats[t]['games']>0 else 0.0
        def get_form(t): return sum(stats[t]['last_5'])
        features_list.append({
            'home_attack': get_avg(h, 'goals_scored'), 'away_defense': get_avg(a, 'goals_conceded'),
            'away_attack': get_avg(a, 'goals_scored'), 'home_defense': get_avg(h, 'goals_conceded'),
            'home_form_val': get_form(h), 'away_form_val': get_form(a),
            'diff_points': stats[h]['points'] - stats[a]['points']
        })
        if row['status'] in ['FT', 'AET', 'PEN'] and pd.notna(row['home_goals']):
            gh, ga = int(row['home_goals']), int(row['away_goals'])
            stats[h]['games']+=1; stats[a]['games']+=1
            stats[h]['goals_scored']+=gh; stats[h]['goals_conceded']+=ga
            stats[a]['goals_scored']+=ga; stats[a]['goals_conceded']+=gh
            if gh>ga: ph, pa = 3,0
            elif ga>gh: ph, pa = 0,3
            else: ph, pa = 1,1
            stats[h]['points']+=ph; stats[a]['points']+=pa
            stats[h]['last_5'].append(ph); stats[a]['last_5'].append(pa)
            if len(stats[h]['last_5'])>5: stats[h]['last_5'].pop(0)
            if len(stats[a]['last_5'])>5: stats[a]['last_5'].pop(0)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(features_list)], axis=1)

def sanitize_record(record):
    new_record = {}
    for key, value in record.items():
        if isinstance(value, list): new_record[key] = [sanitize_record(v) if isinstance(v, dict) else v for v in value]
        elif isinstance(value, dict): new_record[key] = sanitize_record(value)
        elif isinstance(value, (np.integer, np.int64)): new_record[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)): new_record[key] = float(value)
        elif pd.isna(value): new_record[key] = None
        else: new_record[key] = value
    return new_record

def rodar_robo():
    if not firebase_admin._apps: return
    batch = db.batch()
    count_saved = 0
    count_batch = 0
    
    for league_id, league_name in LEAGUES.items():
        print(f"\n--- Processando: {league_name} ---", flush=True)
        
        # Pausa estratégica para evitar rate limit
        time.sleep(7) 
        
        coletar_e_salvar_tabela(league_id, league_name)
        
        time.sleep(1) 
        df = coletar_campeonato(league_id, league_name)
        
        if df.empty: continue
        
        df_enriched = engenharia_de_features(df)
        df_treino = df_enriched.dropna(subset=['result'])
        
        model_match = None
        model_goals = None
        
        if len(df_treino) > 10:
            X = df_treino[['diff_points', 'home_form_val', 'away_form_val', 'home_attack', 'home_defense', 'away_attack', 'away_defense']]
            
            # IA 1: Previsão de Vencedor
            y_match = df_treino['result'].astype(int)
            model_match = RandomForestClassifier(n_estimators=50, random_state=42)
            model_match.fit(X, y_match)
            
            # IA 2: Previsão de Golos (Over 2.5)
            # Criamos target: 1 se > 2.5, 0 se <= 2.5
            y_goals = df_treino['over_25'].astype(int)
            model_goals = RandomForestClassifier(n_estimators=50, random_state=42)
            model_goals.fit(X, y_goals)

        print(f"   Salvando {len(df_enriched)} jogos...", flush=True)
        
        for index, row in df_enriched.iterrows():
            probs = {'home': 33, 'draw': 34, 'away': 33}
            probs_goals = {'over': 50, 'under': 50} # Padrão
            insight = "Aguardando."
            
            if model_match and model_goals:
                try:
                    feats = [[row['diff_points'], row['home_form_val'], row['away_form_val'], row['home_attack'], row['home_defense'], row['away_attack'], row['away_defense']]]
                    if not np.isnan(feats).any():
                        # Vencedor
                        p_match = model_match.predict_proba(feats)[0]
                        probs = {'home': int(p_match[1]*100), 'draw': int(p_match[0]*100), 'away': int(p_match[2]*100)}
                        
                        # Golos
                        p_goals = model_goals.predict_proba(feats)[0]
                        # p_goals[1] é probabilidade de Over 2.5
                        probs_goals = {'under': int(p_goals[0]*100), 'over': int(p_goals[1]*100)}
                        
                        if p_match[1] > 0.6: insight = f"{row['home_team']} favorito."
                        elif p_match[2] > 0.6: insight = f"{row['away_team']} favorito."
                        elif p_goals[1] > 0.65: insight = "Expectativa de muitos gols (Over 2.5)."
                        else: insight = "Jogo equilibrado."
                except: pass

            h2h_data = calcular_historico_h2h(df_enriched, row['home_team'], row['away_team'], row['date'])

            doc_ref = db.collection('games').document(row['id'])
            
            dados_raw = {
                'id': row['id'],
                'leagueId': league_id,
                'leagueName': league_name,
                'round': int(row['round']) if pd.notna(row['round']) else 0,
                'roundLabel': str(row['round_label']),
                'homeTeam': str(row['home_team']), 'awayTeam': str(row['away_team']),
                'homeLogo': str(row['home_logo']), 'awayLogo': str(row['away_logo']),
                'homeScore': int(row['home_goals']) if pd.notna(row['home_goals']) else None,
                'awayScore': int(row['away_goals']) if pd.notna(row['away_goals']) else None,
                'date': row['date'].strftime("%d/%m %H:%M"),
                'venue': str(row['venue']),
                'probs': probs,
                'probs_goals': probs_goals, # Campo novo!
                'h2h': h2h_data,
                'stats': {
                    'homeAttack': row['home_attack'], 'homeDefense': row['home_defense'], 
                    'awayAttack': row['away_attack'], 'awayDefense': row['away_defense'],
                    'isMock': False
                },
                'insight': insight,
                'timestamp': int(row['date'].timestamp() * 1000),
                'status': row['status']
            }
            batch.set(doc_ref, sanitize_record(dados_raw))
            count_saved += 1
            count_batch += 1
            if count_batch >= 400:
                batch.commit(); batch = db.batch(); count_batch = 0
                print(f"   ... lote salvo.", flush=True)

    if count_batch > 0: batch.commit()
    print(f"\n✅ FINALIZADO! Jogos com Previsão de Golos atualizados.", flush=True)

if __name__ == "__main__":
    rodar_robo()
