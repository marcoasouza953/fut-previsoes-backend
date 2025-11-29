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
# 1. CONFIGURAÇÕES (NOVA API)
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

SEASON_TARGET = datetime.datetime.now().year

print(f"⚙️ NOVO ROBÔ INICIADO (API: football-data.org)", flush=True)

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
    
    url = f"https://api.football-data.org/v4/competitions/{league_id}/matches"
    headers = {'X-Auth-Token': API_KEY}
    params = {"season": SEASON_TARGET} 
    
    try:
        resp = requests.get(url, headers=headers, params=params)
        
        if resp.status_code == 403:
            print("      ❌ Erro 403: Chave inválida ou bloqueada.", flush=True)
            return pd.DataFrame()
        if resp.status_code == 429:
            print("      ⚠️ Erro 429: Muitas requisições. Espere um pouco.", flush=True)
            return pd.DataFrame()
            
        data = resp.json()
        
        if 'matches' not in data:
            print(f"      ⚠️ Nenhum jogo encontrado ou erro: {data}", flush=True)
            return pd.DataFrame()
            
        jogos = []
        for item in data['matches']:
            rodada_num = item.get('matchday', 0)
            home_team = item['homeTeam']['name']
            away_team = item['awayTeam']['name']
            home_logo = item['homeTeam'].get('crest', '')
            away_logo = item['awayTeam'].get('crest', '')
            
            score_h = item['score']['fullTime']['home']
            score_a = item['score']['fullTime']['away']
            status_api = item['status']
            
            if status_api == 'FINISHED' and score_h is not None and score_a is not None:
                status = 'FT'
                result = 1 if score_h > score_a else (2 if score_a > score_h else 0)
            else:
                status = 'NS'
                result = None
                
            jogos.append({
                'id': str(item['id']),
                'league_id': league_id,
                'date': item['utcDate'],
                'home_team': home_team,
                'away_team': away_team,
                'home_logo': home_logo,
                'away_logo': away_logo,
                'home_goals': score_h,
                'away_goals': score_a,
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

        if row['status'] == 'FT' and pd.notna(row['home_goals']):
            gh = int(row['home_goals'])
            ga = int(row['away_goals'])
            
            stats[h]['games'] += 1; stats[a]['games'] += 1
            stats[h]['goals_scored'] += gh; stats[h]['goals_conceded'] += ga
            stats[a]['goals_scored'] += ga; stats[a]['goals_conceded'] += gh
            
            res = 1 if gh > ga else (2 if ga > gh else 0)
            pts_h = 3 if res == 1 else (1 if res == 0 else 0)
            pts_a = 3 if res == 2 else (1 if res == 0 else 0)
            
            stats[h]['points'] += pts_h; stats[a]['points'] += pts_a
            stats[h]['last_5'].append(pts_h)
            if len(stats[h]['last_5']) > 5: stats[h]['last_5'].pop(0)
            stats[a]['last_5'].append(pts_a)
            if len(stats[a]['last_5']) > 5: stats[a]['last_5'].pop(0)

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(features_list)], axis=1)

def sanitize_record(record):
    """
    Função crucial: Converte tipos Numpy/Pandas para tipos nativos do Python
    para evitar erros no Firebase.
    """
    new_record = {}
    for key, value in record.items():
        if isinstance(value, (np.integer, np.int64, np.int32)):
            new_record[key] = int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            new_record[key] = float(value)
        elif isinstance(value, (np.ndarray, list)):
            new_record[key] = [sanitize_record(v) if isinstance(v, dict) else v for v in value]
        elif isinstance(value, dict):
            new_record[key] = sanitize_record(value)
        elif pd.isna(value):  # Trata NaN e NaT
            new_record[key] = None
        else:
            new_record[key] = value
    return new_record

def rodar_robo_novo():
    if not firebase_admin._apps: return
    batch = db.batch()
    count_total = 0
    count_batch = 0
    
    for league_id, league_name in LEAGUES.items():
        print(f"\n--- Processando: {league_name} ---", flush=True)
        
        try:
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
                        # Tratar NaNs antes de prever
                        if not np.isnan(feats).any():
                            p = model.predict_proba(feats)[0]
                            probs = {'home': int(p[1]*100), 'draw': int(p[0]*100), 'away': int(p[2]*100)}
                            
                            if p[1] > 0.55: insight = f"{row['home_team']} favorito em casa."
                            elif p[2] > 0.55: insight = f"{row['away_team']} favorito fora."
                            else: insight = "Jogo equilibrado."
                    except Exception as e:
                        # Erros de previsão não devem parar o script
                        pass

                ts = int(row['date'].timestamp() * 1000)
                date_fmt = row['date'].strftime("%d/%m %H:%M")
                
                doc_ref = db.collection('games').document(row['id'])
                
                # Monta dicionário bruto
                dados_raw = {
                    'id': row['id'],
                    'leagueId': league_id,
                    'leagueName': league_name,
                    'round': int(row['round']) if pd.notna(row['round']) else 0,
                    'roundLabel': str(row['round_label']),
                    'homeTeam': str(row['home_team']), 'awayTeam': str(row['away_team']),
                    'homeLogo': str(row['home_logo']), 'awayLogo': str(row['away_logo']),
                    'homeScore': row['home_goals'], # Deixe o sanitizer tratar
                    'awayScore': row['away_goals'],
                    'date': date_fmt,
                    'venue': str(row['venue']),
                    'probs': probs,
                    'stats': {
                        'homeAttack': row['home_attack'], 
                        'homeDefense': row['home_defense'], 
                        'awayAttack': row['away_attack'], 
                        'awayDefense': row['away_defense'],
                        'isMock': False
                    },
                    'insight': insight,
                    'timestamp': ts,
                    'status': row['status']
                }
                
                # SANITIZAÇÃO (A CORREÇÃO PRINCIPAL)
                dados_safe = sanitize_record(dados_raw)
                
                batch.set(doc_ref, dados_safe)
                count_total += 1
                count_batch += 1
                
                if count_batch >= 400:
                    print(f"   ... comitando lote de 400 jogos...", flush=True)
                    batch.commit()
                    batch = db.batch()
                    count_batch = 0

        except Exception as e:
            print(f"❌ Erro ao processar liga {league_name}: {e}", flush=True)
            traceback.print_exc()

    # Comita o restante
    if count_batch > 0:
        batch.commit()
        print(f"   ... último lote salvo.", flush=True)

    if count_total > 0:
        print(f"\n✅ SUCESSO! Total de jogos salvos com NOVA API: {count_total}", flush=True)
    else:
        print("\n⚠️ Nenhum jogo salvo. Verifique logs acima.", flush=True)

if __name__ == "__main__":
    try:
        rodar_robo_novo()
    except Exception as e:
        print(f"🔥 ERRO FATAL: {e}")
        traceback.print_exc()
        # Não usaremos sys.exit(1) para garantir que os logs sejam impressos até o fim
