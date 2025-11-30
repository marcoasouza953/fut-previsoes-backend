import os
import json
import sys
import traceback
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
# 1. CONFIGURAÇÕES
# -------------------------------------------------------------------------
API_KEY = os.environ.get("FOOTBALL_API_KEY", "SUA_NOVA_API_KEY_AQUI") 

LEAGUES = {
    '2013': 'Brasileirão Série A',
    '2021': 'Premier League (ING)',
    '2014': 'La Liga (ESP)',
    '2001': 'Champions League',
    '2019': 'Serie A (ITA)',
    '2002': 'Bundesliga (ALE)'
}

# --- CORREÇÃO: Forçar 2023 para garantir dados na API Grátis ---
SEASON_TARGET = "2023" 
# ---------------------------------------------------------------

print(f"⚙️ ROBÔ INICIADO: Baixando dados de {SEASON_TARGET}", flush=True)

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
    url = f"https://api.football-data.org/v4/competitions/{league_id}/standings"
    headers = {'X-Auth-Token': API_KEY}
    params = {"season": SEASON_TARGET}
    
    try:
        resp = requests.get(url, headers=headers, params=params)
        data = resp.json()
        
        if 'standings' not in data: 
            print("      ⚠️ Tabela não encontrada.", flush=True)
            return

        standings_data = []
        for group in data['standings']:
            if group['type'] == 'TOTAL':
                for team_rank in group['table']:
                    standings_data.append({
                        'rank': team_rank['position'],
                        'teamName': team_rank['team']['name'],
                        'teamLogo': team_rank['team'].get('crest', ''),
                        'points': team_rank['points'],
                        'goalsDiff': team_rank['goalDifference'],
                        'played': team_rank['playedGames'],
                        'win': team_rank['won'],
                        'draw': team_rank['draw'],
                        'lose': team_rank['lost'],
                        'form': team_rank.get('form', ''),
                        'group': group.get('group', 'Único')
                    })
        
        if firebase_admin._apps and standings_data:
            doc_ref = db.collection('standings').document(str(league_id))
            doc_ref.set({
                'leagueName': league_name,
                'updatedAt': int(datetime.datetime.now().timestamp() * 1000),
                'table': standings_data
            })
            print(f"      ✅ Tabela salva ({len(standings_data)} times).", flush=True)

    except Exception as e:
        print(f"      ❌ Erro tabela: {e}", flush=True)

def coletar_campeonato(league_id, league_name):
    print(f"   -> Baixando Jogos de {league_name}...", flush=True)
    url = f"https://api.football-data.org/v4/competitions/{league_id}/matches"
    headers = {'X-Auth-Token': API_KEY}
    params = {"season": SEASON_TARGET} 
    
    try:
        resp = requests.get(url, headers=headers, params=params)
        data = resp.json()
        
        if "errors" in data and data["errors"]:
            print(f"      ❌ Erro API: {data['errors']}", flush=True)
            return pd.DataFrame()
            
        jogos = []
        if 'matches' in data:
            for item in data['matches']:
                rodada_num = item.get('matchday', 0)
                home = item['homeTeam']['name']
                away = item['awayTeam']['name']
                home_logo = item['homeTeam'].get('crest', '')
                away_logo = item['awayTeam'].get('crest', '')
                score_h = item['score']['fullTime']['home']
                score_a = item['score']['fullTime']['away']
                status_api = item['status']
                
                status = 'NS'
                result = None
                if status_api == 'FINISHED':
                    status = 'FT'
                    if score_h is not None and score_a is not None:
                        result = 1 if score_h > score_a else (2 if score_a > score_h else 0)
                
                jogos.append({
                    'id': str(item['id']),
                    'league_id': league_id,
                    'date': item['utcDate'],
                    'home_team': home, 'away_team': away,
                    'home_logo': home_logo, 'away_logo': away_logo,
                    'home_goals': score_h, 'away_goals': score_a,
                    'result': result,
                    'venue': "Estádio",
                    'round': rodada_num,
                    'round_label': f"Rodada {rodada_num}",
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
        historico.append({
            'date': row['date'].strftime("%d/%m/%y"),
            'home': row['home_team'], 'away': row['away_team'],
            'score': f"{int(row['home_goals'])} - {int(row['away_goals'])}",
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
        
        if row['status'] == 'FT' and pd.notna(row['home_goals']):
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
        
        coletar_e_salvar_tabela(league_id, league_name)
        
        df = coletar_campeonato(league_id, league_name)
        if df.empty: continue
        
        df_enriched = engenharia_de_features(df)
        
        df_treino = df_enriched[df_enriched['status'] == 'FT']
        model = None
        if len(df_treino) > 10:
            X = df_treino[['diff_points', 'home_form_val', 'away_form_val', 'home_attack', 'home_defense', 'away_attack', 'away_defense']]
            y = df_treino['result'].astype(int)
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X, y)

        print(f"   Salvando {len(df_enriched)} jogos...", flush=True)
        
        for index, row in df_enriched.iterrows():
            probs = {'home': 33, 'draw': 34, 'away': 33}
            insight = "Aguardando."
            
            if model:
                try:
                    feats = [[row['diff_points'], row['home_form_val'], row['away_form_val'], row['home_attack'], row['home_defense'], row['away_attack'], row['away_defense']]]
                    if not np.isnan(feats).any():
                        p = model.predict_proba(feats)[0]
                        probs = {'home': int(p[1]*100), 'draw': int(p[0]*100), 'away': int(p[2]*100)}
                        if p[1] > 0.6: insight = f"{row['home_team']} favorito."
                        elif p[2] > 0.6: insight = f"{row['away_team']} favorito."
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
                'homeScore': row['home_goals'], 'awayScore': row['away_goals'],
                'date': row['date'].strftime("%d/%m %H:%M"),
                'venue': "Estádio",
                'probs': probs,
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
    print(f"\n✅ SUCESSO! Banco de dados preenchido com {count_saved} jogos.", flush=True)

if __name__ == "__main__":
    rodar_robo()
