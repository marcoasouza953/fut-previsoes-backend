import os
import json
import requests
import pandas as pd
import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------------------------
# CONFIGURAÇÕES (Lê das Variáveis de Ambiente para segurança)
# -------------------------------------------------------------------------

# 1. API Key (Tenta pegar do ambiente, senão usa a string hardcoded para testes locais)
API_KEY = os.environ.get("FOOTBALL_API_KEY", "SUA_API_KEY_AQUI") 

# 2. Configurações da Liga
LEAGUE_ID = "71"  # Brasileirão Série A
SEASON_TRAIN = "2023"
SEASON_CURRENT = "2024"

# -------------------------------------------------------------------------
# CONEXÃO COM O FIREBASE (Híbrida: Arquivo ou Variável de Ambiente)
# -------------------------------------------------------------------------
print("1. Conectando ao Firebase...")

if not firebase_admin._apps:
    # Tenta ler a chave secreta guardada na variável de ambiente (Método Nuvem)
    firebase_creds_str = os.environ.get("FIREBASE_CREDENTIALS")
    
    if firebase_creds_str:
        print("☁️ Usando credenciais da Nuvem (Env Var)")
        cred_dict = json.loads(firebase_creds_str)
        cred = credentials.Certificate(cred_dict)
    else:
        # Fallback: Tenta ler do arquivo local (Método Colab/PC)
        # Substitua pelo nome do seu arquivo se for rodar localmente
        local_key_path = "firebase_key.json" 
        if os.path.exists(local_key_path):
            print(f"💻 Usando arquivo local: {local_key_path}")
            cred = credentials.Certificate(local_key_path)
        else:
            raise Exception("❌ ERRO: Nenhuma credencial do Firebase encontrada (Nem Env Var, nem Arquivo).")

    firebase_admin.initialize_app(cred)

db = firestore.client()
print("✅ Conectado ao banco de dados!")

# -------------------------------------------------------------------------
# LÓGICA DE INTELIGÊNCIA ARTIFICIAL (O Robô)
# -------------------------------------------------------------------------

def coletar_dados_treino():
    print(f"\n2. Baixando histórico de {SEASON_TRAIN}...")
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {'x-apisports-key': API_KEY}
    params = {"league": LEAGUE_ID, "season": SEASON_TRAIN}
    
    resp = requests.get(url, headers=headers, params=params)
    data = resp.json()
    
    jogos = []
    for item in data['response']:
        if item['fixture']['status']['short'] != 'FT': continue
        
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
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date')

def treinar_modelo(df):
    print("3. Treinando a IA...")
    # Recalcula a tabela de pontos rodada a rodada
    team_stats = {team: 0 for team in set(df['home_team']).union(set(df['away_team']))}
    home_pts, away_pts = [], []
    
    for index, row in df.iterrows():
        home_pts.append(team_stats.get(row['home_team'], 0))
        away_pts.append(team_stats.get(row['away_team'], 0))
        
        if row['result'] == 1: team_stats[row['home_team']] += 3
        elif row['result'] == 2: team_stats[row['away_team']] += 3
        else:
            team_stats[row['home_team']] += 1
            team_stats[row['away_team']] += 1
            
    df['home_points_pre'] = home_pts
    df['away_points_pre'] = away_pts
    df['points_diff'] = df['home_points_pre'] - df['away_points_pre']
    
    X = df[['home_points_pre', 'away_points_pre', 'points_diff']]
    y = df['result']
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model, team_stats

def atualizar_previsoes():
    # 1. Prepara dados
    df_train = coletar_dados_treino()
    model, team_stats = treinar_modelo(df_train)
    
    # 2. Busca jogos futuros
    print(f"\n4. Buscando jogos reais ({SEASON_CURRENT})...")
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {'x-apisports-key': API_KEY}
    # Busca próximos 10 jogos "Not Started" (NS)
    params = {"league": LEAGUE_ID, "season": SEASON_CURRENT, "next": "10"} 
    
    resp = requests.get(url, headers=headers, params=params)
    data_next = resp.json()
    
    jogos_futuros = []
    usando_simulacao = False
    
    if data_next['results'] > 0:
        print(f"✅ Encontrados {data_next['results']} jogos futuros.")
        jogos_futuros = data_next['response']
    else:
        print("⚠️ Nenhum jogo agendado encontrado. Usando SIMULAÇÃO.")
        usando_simulacao = True
        # Usa os últimos 5 jogos do treino como simulação
        jogos_futuros = df_train.tail(5).to_dict('records')

    # 3. Faz previsões e salva
    batch = db.batch()
    print("\n5. Salvando no Firebase...")
    
    for item in jogos_futuros:
        if usando_simulacao:
            # Lógica de Simulação
            game_id = f"sim_{item['date'].strftime('%Y%m%d')}_{item['home_team'][:3]}"
            home = item['home_team']
            away = item['away_team']
            date_str = item['date'].strftime("%d/%m %H:%M")
            timestamp = int(item['date'].timestamp() * 1000)
            venue = "Simulação Arena"
            features = [[item['home_points_pre'], item['away_points_pre'], item['points_diff']]]
        else:
            # Lógica Real
            game_id = str(item['fixture']['id'])
            home = item['teams']['home']['name']
            away = item['teams']['away']['name']
            venue = item['fixture']['venue']['name']
            dt_obj = datetime.datetime.fromisoformat(item['fixture']['date'].replace('Z', '+00:00'))
            date_str = dt_obj.strftime("%d/%m %H:%M")
            timestamp = int(dt_obj.timestamp() * 1000)
            
            # Pega pontos atuais do dicionário calculado no treino
            pts_h = team_stats.get(home, 0)
            pts_a = team_stats.get(away, 0)
            features = [[pts_h, pts_a, pts_h - pts_a]]

        # Previsão
        probs = model.predict_proba(features)[0]
        
        doc_ref = db.collection('games').document(game_id)
        dados = {
            'id': game_id,
            'homeTeam': home,
            'awayTeam': away,
            'date': date_str,
            'venue': venue,
            'homeForm': "?????",
            'awayForm': "?????",
            'probs': {
                'home': int(probs[1] * 100),
                'draw': int(probs[0] * 100),
                'away': int(probs[2] * 100)
            },
            'insight': "Atualizado automaticamente pelo Robô Python.",
            'timestamp': timestamp,
            'is_simulation': usando_simulacao
        }
        batch.set(doc_ref, dados)
        
    batch.commit()
    print("✅ CICLO COMPLETO! Previsões atualizadas.")

if __name__ == "__main__":
    atualizar_previsoes()
