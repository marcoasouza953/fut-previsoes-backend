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
# IMPORTANTE: Verifique se sua chave está aqui!
API_KEY = os.environ.get("FOOTBALL_API_KEY", "SUA_API_KEY_AQUI") 

# Se quiser testar outro ano, mude aqui (ex: "2024" ou "2025")
SEASON_TARGET = "2023" 

LEAGUES = {
    '71': 'Brasileirão Série A',
    '13': 'Copa Libertadores',
    '2': 'Champions League',
    '11': 'Copa Sul-Americana',
    '39': 'Premier League (ING)',
    '140': 'La Liga (ESP)'
}

print(f"⚙️ MODO DIAGNÓSTICO: Verificando {len(LEAGUES)} ligas na temporada {SEASON_TARGET}")
print(f"🔑 Usando chave (início): {API_KEY[:5]}...")

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
            print("⚠️ AVISO: Sem credenciais do Firebase. O script vai rodar mas NÃO vai salvar.")

if firebase_admin._apps:
    db = firestore.client()
    print("✅ Conectado ao banco de dados!")

# -------------------------------------------------------------------------
# LÓGICA DE DADOS
# -------------------------------------------------------------------------

def converter_rodada_para_numero(texto_rodada):
    texto = str(texto_rodada).lower()
    if "final" in texto and "semi" not in texto and "quarter" not in texto: return 50
    if "semi" in texto: return 49
    if "quarter" in texto: return 48
    if "16" in texto or "8th" in texto: return 47
    numeros = re.findall(r'\d+', texto)
    return int(numeros[0]) if numeros else 0

def coletar_campeonato(league_id, league_name):
    print(f"   -> Baixando {league_name} (ID {league_id})...")
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {'x-apisports-key': API_KEY}
    params = {"league": league_id, "season": SEASON_TARGET} 
    
    try:
        resp = requests.get(url, headers=headers, params=params)
        data = resp.json()
        
        # --- BLOCO DE DIAGNÓSTICO ---
        if "errors" in data and data["errors"]:
            print(f"      ❌ ERRO DA API: {data['errors']}")
            # Verifica erros comuns
            err_str = str(data['errors'])
            if "key" in err_str.lower(): print("      💡 DICA: Sua API Key parece inválida ou ausente.")
            if "limit" in err_str.lower(): print("      💡 DICA: Você atingiu o limite diário (100/100).")
            return pd.DataFrame()
            
        if data['results'] == 0:
            print(f"      ⚠️ API retornou 0 jogos. A temporada {SEASON_TARGET} existe para esta liga?")
            return pd.DataFrame()
        # ---------------------------

    except Exception as e:
        print(f"      ❌ Erro Crítico de Conexão: {e}")
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
                'league_id': league_id,
                'date': item['fixture']['date'],
                'home_team': item['teams']['home']['name'],
                'away_team': item['teams']['away']['name'],
                'home_goals': home,
                'away_goals': away,
                'result': result,
                'venue': item['fixture']['venue']['name'],
                'round': rodada_num,
                'round_label': rodada_str,
                'status': status
            })
    
    df = pd.DataFrame(jogos)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        print(f"      ✅ Sucesso: {len(df)} jogos baixados.")
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
    batch = None
    if firebase_admin._apps:
        batch = db.batch()
    
    count_total = 0
    
    for league_id, league_name in LEAGUES.items():
        print(f"\n--- Processando: {league_name} ---")
        
        df = coletar_campeonato(league_id, league_name)
        if df.empty: continue
        
        df_enriched = engenharia_de_features(df)
        
        df_treino = df_enriched.dropna(subset=['result'])
        model = None
        if len(df_treino) > 10:
            X = df_treino[['diff_points', 'home_form_val', 'away_form_val', 'home_attack', 'home_defense', 'away_attack', 'away_defense']]
            y = df_treino['result'].astype(int)
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X, y)

        if not firebase_admin._apps:
            print("   (Modo Teste: Não salvando no Firebase)")
            continue

        print(f"   Salvando {len(df_enriched)} jogos no Firebase...")
        
        for index, row in df_enriched.iterrows():
            probs = {'home': 33, 'draw': 34, 'away': 33}
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
                'leagueId': league_id,
                'leagueName': league_name,
                'round': int(row['round']),
                'roundLabel': row['round_label'],
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
                print(f"   ... {count_total} jogos salvos até agora...")

    if batch and count_total > 0:
        batch.commit()
        print(f"\n✅ SUCESSO FINAL! Total de jogos salvos: {count_total}")
    elif count_total == 0:
        print("\n❌ FALHA: Nenhum jogo foi salvo. Verifique os erros acima.")

if __name__ == "__main__":
    rodar_robo_multi_liga()
