import requests
import os
import json
import datetime

# ----------------------------------------------------------
# COLOQUE SUA CHAVE AQUI PARA TESTAR
# ----------------------------------------------------------
# Se estiver no GitHub Actions, ele pega do ambiente.
# Se estiver no PC/Colab, substitua o texto abaixo pela chave.
API_KEY = os.environ.get("FOOTBALL_API_KEY", "SUA_API_KEY_AQUI") 

LEAGUE_ID = "71"  # Brasileirão
SEASON = "2025"   # Ano Atual

print("--- INICIANDO DIAGNÓSTICO ---")
print(f"Chave usada (primeiros 5 dígitos): {API_KEY[:5]}...")

# 1. TESTE DE CONTA (Verifica se a chave é válida e se tem cota)
print("\n1. Verificando Status da Conta...")
url_status = "https://v3.football.api-sports.io/status"
headers = {'x-apisports-key': API_KEY}

try:
    resp = requests.get(url_status, headers=headers)
    data = resp.json()
    
    if "errors" in data and data["errors"]:
        print("❌ ERRO CRÍTICO NA CONTA:")
        print(json.dumps(data["errors"], indent=4))
    else:
        conta = data['response']['account']
        print(f"✅ Conta: {conta['firstname']} {conta['lastname']}")
        print(f"✅ Email: {conta['email']}")
        print(f"📊 Requisições hoje: {data['response']['requests']['current']} / {data['response']['requests']['limit_day']}")
        
        if data['response']['requests']['current'] >= data['response']['requests']['limit_day']:
            print("⚠️ ALERTA: Você atingiu o limite de 100 requisições hoje! A API não vai retornar mais jogos.")

except Exception as e:
    print(f"❌ Erro de conexão: {e}")

# 2. TESTE DE JOGOS (Tenta buscar jogos reais)
print(f"\n2. Buscando jogos do Brasileirão ({SEASON})...")
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
        print("   - O campeonato acabou.")
        print("   - A tabela ainda não foi sorteada.")
        print("   - Erro na API (veja resposta bruta abaixo):")
        print(data)

except Exception as e:
    print(f"❌ Erro ao buscar jogos: {e}")

print("\n--- FIM DO DIAGNÓSTICO ---")