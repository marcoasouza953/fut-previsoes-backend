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
# 1. CONFIGURAÇÕES
# -------------------------------------------------------------------------
API_KEY = os.environ.get("FOOTBALL_API_KEY", "SUA_API_KEY_AQUI") 

# Lista de Campeonatos (ID da API-Football)
# Você pode adicionar mais aqui!
LEAGUES = {
    '71': 'Brasileirão Série A',
    '13': 'Copa Libertadores',
    '2': 'Champions League',
    '11': 'Copa Sul-Americana',
    '39': 'Premier League (ING)',
    '140': 'La Liga (ESP)'
}

# Vamos focar na temporada passada/atual completa para ter dados
SEASON_TARGET = "2023" 

print(f"⚙️ MODO MULTI-LIGA: Processando {len(LEAGUES)} campeonatos...")

# -------------------------------------------------------------------------
# CONEXÃO FIREBASE
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
            cred = credentials.Certificate(local_key_path)
            firebase_admin.initialize_app(cred)
        else:
            print("⚠️ AVISO: Sem credenciais.")

if firebase_admin._apps:
    db = firestore.client()

# -------------------------------------------------------------------------
# LÓGICA DE DADOS
# -------------------------------------------------------------------------

def converter_rodada_para_numero(texto_rodada):
    """
    Transforma nomes de fases em números para o App ordenar.
    Ex: "Regular Season - 5" -> 5
    Ex: "Final" -> 50
    """
    texto = str(texto_rodada).lower()
    
    # Mapeamento para Copas (Libertadores/Champions)
    if "final" in texto and "semi" not in texto and "quarter" not in texto and "8" not in texto: return 50
    if "semi" in texto: return 49
    if "quarter" in texto: return 48
    if "16" in texto or "8th" in texto: return 47 # Oitavas
    if "32" in texto: return 46
    
    # Tenta extrair número normal (Fase de Grupos ou Pontos Corridos)
    numeros = re.findall(r'\d+', texto)
    if numeros:
        return int(numeros[0])
        
    return 0 # Desconhecido

def coletar_campeonato(league_id):
    print(f"   -> Baixando Liga ID {league_id}...")
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {'x-apisports-key': API_KEY}
    params = {"league": league_id, "season": SEASON_TARGET} 
    
    try:
        resp = requests.get(url, headers=headers, params=params)
        data = resp.json()
    except Exception as e:
        print(f"❌ Erro API: {e}")
        return pd.DataFrame()
    
    jogos = []
    if 'response' in data:
        for item in data['response']:
            rodada_str = item['league']['round']
            rodada_num = converter_rodada_para_numero(rodada_str)
            
            home = item['goals']['home']
            away = item['goals']['away']
            
            status = item['fixture']['status']['short']
            result = None
            if status == 'FT' and home is not None and away is not None:
                result = 1 if home > away else (2 if away > home else 0)
            
            jogos.append({
                'id': str(item['fixture']['id']),
                'league_id': league_id, # Importante para filtrar no App
                'date': item['fixture']['date'],
                'home_team': item['teams']['home']['name'],
                'away_team': item['teams']['away']['name'],
                'home_goals': home,
                'away_goals': away,
                'result': result,
                'venue': item['fixture']['venue']['name'],
                'round': rodada_num,
                'round_label': rodada_str, # Nome original (ex: "Final")
                'status': status
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

        if row['result'] is not None:
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

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(features_list)], axis=1)

def rodar_robo_multi_liga():
    if not firebase_admin._apps: return
    batch = db.batch()
    count_total = 0
    
    # Loop por cada campeonato
    for league_id, league_name in LEAGUES.items():
        print(f"\n--- Processando: {league_name} ---")
        
        df = coletar_campeonato(league_id)
        if df.empty: continue
        
        df_enriched = engenharia_de_features(df)
        
        # Treina IA específica para esta liga (melhor precisão)
        df_treino = df_enriched.dropna(subset=['result'])
        if len(df_treino) > 10:
            X = df_treino[['diff_points', 'home_form_val', 'away_form_val', 'home_attack', 'home_defense', 'away_attack', 'away_defense']]
            y = df_treino['result'].astype(int)
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X, y)
        else:
            print("   ⚠️ Poucos dados para treinar IA nesta liga.")
            model = None

        print(f"   Salvando {len(df_enriched)} jogos...")
        
        for index, row in df_enriched.iterrows():
            # Probabilidades
            probs = {'home': 33, 'draw': 34, 'away': 33} # Padrão
            insight = "Dados insuficientes."
            
            if model:
                feats = [[
                    row['diff_points'], row['home_form_val'], row['away_form_val'],
                    row['home_attack'], row['home_defense'], row['away_attack'], row['away_defense']
                ]]
                p = model.predict_proba(feats)[0]
                probs = {'home': int(p[1]*100), 'draw': int(p[0]*100), 'away': int(p[2]*100)}
                
                if p[1] > 0.6: insight = f"{row['home_team']} é muito favorito em casa."
                elif p[2] > 0.6: insight = f"{row['away_team']} deve vencer fora."
                else: insight = "Jogo muito equilibrado."

            ts = int(row['date'].timestamp() * 1000)
            date_fmt = row['date'].strftime("%d/%m")
            
            doc_ref = db.collection('games').document(row['id'])
            
            dados = {
                'id': row['id'],
                'leagueId': league_id,      # Essencial para filtro
                'leagueName': league_name,  # Para mostrar no App
                'round': int(row['round']),
                'roundLabel': row['round_label'], # "Final", "Rodada 1"
                'homeTeam': row['home_team'], 'awayTeam': row['away_team'],
                'homeScore': int(row['home_goals']) if pd.notna(row['home_goals']) else None,
                'awayScore': int(row['away_goals']) if pd.notna(row['away_goals']) else None,
                'date': date_fmt,
                'venue': row['venue'] or "",
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
                'status': row['status']
            }
            batch.set(doc_ref, dados)
            count_total += 1
            
            if count_total % 400 == 0:
                batch.commit()
                batch = db.batch()
                print(f"   ... lote salvo.")

    batch.commit()
    print(f"✅ FINALIZADO! Total de jogos salvos: {count_total}")

if __name__ == "__main__":
    rodar_robo_multi_liga()