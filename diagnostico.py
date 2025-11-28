import requests
import os
import json
import datetime

# Pega a chave dos "Segredos" do GitHub
API_KEY = os.environ.get("FOOTBALL_API_KEY") 
LEAGUE_ID = "71"  # Brasileirão
SEASON = "2025"   # Ano Atual

print("--- A INICIAR DIAGNÓSTICO ---")

if not API_KEY:
    print("❌ ERRO CRÍTICO: Não encontrei a FOOTBALL_API_KEY nas variáveis de ambiente.")
    exit()

print(f"Chave detetada (início): {API_KEY[:4]}...")

# 1. TESTE DE CONTA
print("\n1. A verificar Estado da Conta...")
url_status = "https://v3.football.api-sports.io/status"
headers = {'x-apisports-key': API_KEY}

try:
    resp = requests.get(url_status, headers=headers)
    data = resp.json()
    
    if "errors" in data and data["errors"]:
        print("❌ ERRO NA CONTA:")
        print(json.dumps(data["errors"], indent=4))
    else:
        # Verifica se 'account' existe na resposta
        if 'account' in data.get('response', {}):
            conta = data['response']['account']
            reqs = data['response']['requests']
            print(f"✅ Conta: {conta['firstname']} {conta['lastname']}")
            print(f"📊 Requisições hoje: {reqs['current']} / {reqs['limit_day']}")
            
            if reqs['current'] >= reqs['limit_day']:
                print("⚠️ ALERTA: Atingiu o limite diário! A API não retornará mais jogos.")
        else:
            print("⚠️ Resposta estranha da API (sem dados de conta):")
            print(data)

except Exception as e:
    print(f"❌ Erro de conexão: {e}")

# 2. TESTE DE JOGOS
print(f"\n2. A buscar jogos do Brasileirão ({SEASON})...")
url_fixtures = "https://v3.football.api-sports.io/fixtures"

# Tenta buscar os próximos 10 jogos
params = {"league": LEAGUE_ID, "season": SEASON, "next": "10"}

try:
    resp = requests.get(url_fixtures, headers=headers, params=params)
    data = resp.json()
    
    results = data.get('results', 0)
    print(f"🔍 A API encontrou {results} jogos futuros.")
    
    if results > 0:
        print("✅ JOGOS REAIS ENCONTRADOS:")
        for jogo in data['response']:
            data_jogo = jogo['fixture']['date']
            time_casa = jogo['teams']['home']['name']
            time_fora = jogo['teams']['away']['name']
            print(f"   - {data_jogo}: {time_casa} vs {time_fora}")
    else:
        print("⚠️ A API retornou 0 jogos. Motivos possíveis:")
        print("   - O campeonato acabou ou ainda não começou.")
        print("   - Tente mudar o ANO no script para o ano anterior para testar.")
        print("   - Resposta bruta:")
        print(str(data)[:200]) # Mostra apenas o início para não poluir

except Exception as e:
    print(f"❌ Erro ao buscar jogos: {e}")

print("\n--- FIM DO DIAGNÓSTICO ---")