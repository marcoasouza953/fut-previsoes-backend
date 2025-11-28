import os
import json
import requests
import pandas as pd
import numpy as np
import datetime
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------------------------
# 1. CONFIGURAÇÕES
# -------------------------------------------------------------------------
# Tenta pegar chave do ambiente ou usa a string local
API_KEY = os.environ.get("FOOTBALL_API_KEY", "SUA_API_KEY_AQUI")
LEAGUE_ID = "71"  # Brasileirão

hoje = datetime.datetime.now()
ANO_ATUAL = hoje.year
SEASON_CURRENT = str(ANO_ATUAL)       # Vai ser "2024"
SEASON_TRAIN_HISTORIC = str(ANO_ATUAL - 1) # Vai ser "2023"

print(f"⚙️ Robô Iniciado: {hoje.strftime('%d/%m/%Y')} | Temporada: {SEASON_CURRENT}")

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
# 3. LÓGICA DE DADOS
# -------------------------------------------------------------------------

def consultar_api(endpoint, params):
    """Função auxiliar que trata erros da API"""
    url = f"https://v3.football.api-sports.io/{endpoint}"
    headers = {'x-apisports-key': API_KEY}
    
    try:
        resp = requests.get(url, headers=headers, params=params)
        data = resp.json()
        
        # Verifica erros da API (Limite excedido, chave errada, etc)
        if "errors" in data and data["errors"]:
            print(f"❌ ERRO DA API: {data['errors']}")
            return None
            
        return data
    except Exception as e:
        print(f"❌ Erro de Conexão: {e}")
        return None

def coletar_jogos_realizados(season):
    print(f"   -> Baixando histórico de {season}...")
    # Baixa jogos terminados (FT)
    data = consultar_api("fixtures", {"league": LEAGUE_ID, "season": season, "status": "FT"})
    
    if not data or 'response' not in data: return pd.DataFrame()
    
    jogos = []
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
    
    # Inicializa stats com valores padrão seguros
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

        # Atualiza Stats
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

# -------------------------------------------------------------------------
# 4. FUNÇÃO PRINCIPAL (O Cérebro)
# -------------------------------------------------------------------------
def rodar_robo():
    print("\n2. Treinando Robô...")
    df_historico = coletar_jogos_realizados(SEASON_TRAIN_HISTORIC)
    df_atual = coletar_jogos_realizados(SEASON_CURRENT)
    
    if df_historico.empty and df_atual.empty:
        print("❌ CRÍTICO: Não consegui baixar nenhum jogo histórico. Verifique se sua API Key é válida.")
        return

    df_treino = pd.concat([df_historico, df_atual], ignore_index=True).sort_values('date')
    df_enriched, current_stats = engenharia_de_features(df_treino)
    
    X = df_enriched[['points_diff', 'home_form', 'away_form', 'home_attack', 'away_defense', 'away_attack', 'home_defense']]
    y = df_enriched['result']
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # ---------------------------------------------------------------------
    # ESTRATÉGIA DUPLA DE BUSCA DE JOGOS
    # ---------------------------------------------------------------------
    print(f"\n4. Buscando jogos reais ({SEASON_CURRENT})...")
    
    jogos_futuros = []
    
    # Tentativa 1: Buscar por DATA (Próximos 7 dias)
    data_inicio = datetime.date.today().strftime("%Y-%m-%d")
    data_fim = (datetime.date.today() + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    print(f"   -> Tentativa 1: Intervalo {data_inicio} até {data_fim}")
    
    data_date = consultar_api("fixtures", {"league": LEAGUE_ID, "season": SEASON_CURRENT, "from": data_inicio, "to": data_fim})
    
    if data_date and data_date['results'] > 0:
        jogos_futuros = data_date['response']
        print(f"   ✅ Sucesso! {len(jogos_futuros)} jogos encontrados por data.")
    else:
        # Tentativa 2: Buscar por QUANTIDADE ("Próximos 10", independente da data)
        print("   ⚠️ Nenhum jogo na data. Tentando buscar 'Next 10'...")
        data_next = consultar_api("fixtures", {"league": LEAGUE_ID, "season": SEASON_CURRENT, "next": "10"})
        
        if data_next and data_next['results'] > 0:
            jogos_futuros = data_next['response']
            print(f"   ✅ Sucesso! {len(jogos_futuros)} jogos encontrados pelo método 'Next 10'.")
        else:
             print("   ⚠️ API não retornou NENHUM jogo futuro.")

    # Se mesmo assim não tiver jogos, ativa modo simulação
    usando_simulacao = False
    if not jogos_futuros:
        print("   🔄 Ativando MODO SIMULAÇÃO (Último Recurso)")
        usando_simulacao = True
        last_games = df_treino.tail(5).to_dict('records')
        for lg in last_games:
            jogos_futuros.append({
                'fixture': {'id': f"sim_{lg['date']}", 'date': str(lg['date']), 'venue': {'name': 'Simulação'}, 'status': {'short': 'NS'}},
                'teams': {'home': {'name': lg['home_team']}, 'away': {'name': lg['away_team']}},
                'is_simulation': True
            })

    # ---------------------------------------------------------------------
    # LIMPEZA E SALVAMENTO NO FIREBASE
    # ---------------------------------------------------------------------
    if not firebase_admin._apps: return

    # Passo extra: Limpar simulações antigas se achamos jogos reais
    if not usando_simulacao:
        print("5. Faxina: Removendo jogos simulados antigos...")
        docs = db.collection('games').stream()
        batch_del = db.batch()
        count_del = 0
        for doc in docs:
            d = doc.to_dict()
            # Se o ID começa com 'sim_' ou tem campo is_simulation, deleta
            if str(doc.id).startswith('sim_') or d.get('is_simulation'):
                batch_del.delete(doc.reference)
                count_del += 1
        if count_del > 0:
            batch_del.commit()
            print(f"   🗑️ {count_del} jogos simulados removidos.")

    batch = db.batch()
    print("6. Calculando Previsões e Estatísticas...")
    
    def safe_get_stat(team, metric):
        if team not in current_stats: return 0.0
        s = current_stats[team]
        if metric == 'form': return float(sum(s['last_5']))
        if s['games'] == 0: return 0.0
        return float(s[metric] / s['games'])

    count_saved = 0
    for item in jogos_futuros:
        # Filtra jogos: Se for simulação aceita, senão só aceita NS/TBD
        status = item['fixture']['status']['short']
        if not item.get('is_simulation') and status not in ['NS', 'TBD', 'PST']: 
            continue 

        home = item['teams']['home']['name']
        away = item['teams']['away']['name']
        
        # Pega estatísticas
        h_att = safe_get_stat(home, 'goals_scored')
        h_def = safe_get_stat(home, 'goals_conceded')
        a_att = safe_get_stat(away, 'goals_scored')
        a_def = safe_get_stat(away, 'goals_conceded')
        h_form = safe_get_stat(home, 'form')
        a_form = safe_get_stat(away, 'form')
        
        h_pts = current_stats.get(home, {}).get('points', 0)
        a_pts = current_stats.get(away, {}).get('points', 0)
        
        # Previsão
        features = [[h_pts - a_pts, h_form, a_form, h_att, h_def, a_att, a_def]]
        probs = model.predict_proba(features)[0]
        
        # Insight
        insight = "Jogo equilibrado."
        if probs[1] > 0.6: insight = f"{home} favorito com ataque de {h_att:.1f} gols/jogo."
        elif probs[2] > 0.6: insight = f"{away} perigoso contra defesa do {home}."
        
        game_id = str(item['fixture']['id'])
        doc_ref = db.collection('games').document(game_id)
        
        # Data
        ts = int(datetime.datetime.now().timestamp() * 1000)
        try:
            dt = pd.to_datetime(item['fixture']['date'])
            date_fmt = dt.strftime("%d/%m %H:%M")
            ts = int(dt.timestamp() * 1000)
        except:
            date_fmt = "A Definir"

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
            'probs': {'home': int(probs[1]*100), 'draw': int(probs[0]*100), 'away': int(probs[2]*100)},
            # Stats Forçadas (Garante que nunca ficam vazias)
            'stats': {
                'homeAttack': float(f"{h_att:.2f}"), 
                'homeDefense': float(f"{h_def:.2f}"), 
                'awayAttack': float(f"{a_att:.2f}"), 
                'awayDefense': float(f"{a_def:.2f}"),
                'isMock': False
            },
            'insight': insight,
            'timestamp': ts,
            'is_simulation': usando_simulacao
        }
        batch.set(doc_ref, dados)
        count_saved += 1

    batch.commit()
    print(f"✅ FINALIZADO! {count_saved} jogos salvos no Firebase.")
    if usando_simulacao:
        print("⚠️ NOTA: O Robô usou SIMULAÇÃO. Verifique os erros acima para saber porquê.")

if __name__ == "__main__":
    rodar_robo()