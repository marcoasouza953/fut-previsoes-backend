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

# AGORA BAIXAMOS 2 TEMPORADAS!
SEASONS_TO_FETCH = [2022, 2023]

print(f"⚙️ ROBÔ TURBO INICIADO: Analisando {SEASONS_TO_FETCH}", flush=True)

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
            print("⚠️ AVISO: Sem credenciais.", flush=True)

if firebase_admin._apps:
    db = firestore.client()
    print("✅ Conectado ao banco de dados!", flush=True)

# -------------------------------------------------------------------------
# LÓGICA DE DADOS
# -------------------------------------------------------------------------

def coletar_e_salvar_tabela(league_id, league_name):
    # Usa o ano mais recente para a tabela
    current_season = max(SEASONS_TO_FETCH)
    print(f"   -> Baixando Tabela {current_season} de {league_name}...", flush=True)
    
    url = "https://api.football-data.org/v4/competitions/" + str(league_id) + "/standings"
    headers = {'X-Auth-Token': API_KEY}
    params = {"season": current_season}
    
    try:
        resp = requests.get(url, headers=headers, params=params)
        data = resp.json()
        
        if 'standings' not in data: return

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
            print(f"      ✅ Tabela salva.", flush=True)

    except Exception as e:
        print(f"      ⚠️ Erro tabela: {e}", flush=True)

def coletar_jogos_multi_temporada(league_id, league_name):
    todos_jogos = []
    
    for season in SEASONS_TO_FETCH:
        print(f"   -> Baixando {league_name} ({season})...", flush=True)
        url = f"https://api.football-data.org/v4/competitions/{league_id}/matches"
        headers = {'X-Auth-Token': API_KEY}
        params = {"season": season} 
        
        try:
            resp = requests.get(url, headers=headers, params=params)
            data = resp.json()
            
            if 'matches' in data:
                for item in data['matches']:
                    rodada_num = item.get('matchday', 0)
                    home_team = item['homeTeam']['name']
                    away_team = item['awayTeam']['name']
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
                    
                    todos_jogos.append({
                        'id': str(item['id']),
                        'league_id': league_id,
                        'season': season, # Importante saber o ano
                        'date': item['utcDate'],
                        'home_team': home_team, 'away_team': away_team,
                        'home_logo': home_logo, 'away_logo': away_logo,
                        'home_goals': score_h, 'away_goals': score_a,
                        'result': result,
                        'venue': "Estádio",
                        'round': rodada_num,
                        'round_label': f"Rodada {rodada_num}",
                        'status': status
                    })
        except Exception as e:
            print(f"      ❌ Erro na season {season}: {e}", flush=True)

    df = pd.DataFrame(todos_jogos)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        print(f"      📦 Total acumulado: {len(df)} jogos.", flush=True)
    return df

def calcular_historico_h2h(df_completo, time_a, time_b, data_limite):
    """ Busca confrontos diretos em TODO o histórico baixado (2022+2023) """
    mask = ((df_completo['home_team'] == time_a) & (df_completo['away_team'] == time_b)) | \
           ((df_completo['home_team'] == time_b) & (df_completo['away_team'] == time_a))
    
    # Pega jogos passados terminados
    past_games = df_completo[mask & (df_completo['date'] < data_limite) & (df_completo['status'] == 'FT')]
    
    # Ordena do mais recente para o antigo e pega 5
    ultimos = past_games.sort_values('date', ascending=False).head(5)
    
    historico = []
    for _, row in ultimos.iterrows():
        winner = 'Empate'
        if row['result'] == 1: winner = row['home_team']
        elif row['result'] == 2: winner = row['away_team']
        
        historico.append({
            'date': row['date'].strftime("%d/%m/%y"), # Inclui ano agora
            'home': row['home_team'],
            'away': row['away_team'],
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
        
        # Baixa 2 anos de jogos!
        df = coletar_jogos_multi_temporada(league_id, league_name)
        if df.empty: continue
        
        df_enriched = engenharia_de_features(df)
        
        # Treina IA com dados de 2 anos
        df_treino = df_enriched[df_enriched['status'] == 'FT']
        model = None
        if len(df_treino) > 20:
            X = df_treino[['diff_points', 'home_form_val', 'away_form_val', 'home_attack', 'home_defense', 'away_attack', 'away_defense']]
            y = df_treino['result'].astype(int)
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X, y)

        # Filtra para salvar apenas a temporada atual (para não encher o App de lixo velho)
        # Queremos usar 2022 para cálculo, mas exibir apenas 2023 no App
        latest_season = max(SEASONS_TO_FETCH)
        df_to_save = df_enriched[df_enriched['season'] == latest_season]
        
        print(f"   Salvando {len(df_to_save)} jogos da temporada atual ({latest_season})...", flush=True)
        
        for index, row in df_to_save.iterrows():
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

            # AQUI ESTÁ A MÁGICA: O H2H olha para o df_enriched (que tem 2 anos)
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
                'h2h': h2h_data, # Agora vem recheado!
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
    print(f"\n✅ SUCESSO! Histórico H2H turbinado com 2 anos de dados.", flush=True)

if __name__ == "__main__":
    rodar_robo()
