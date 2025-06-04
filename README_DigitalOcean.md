# Konfiguracja na Digital Ocean

## Sposoby ustawienia klucza API OpenAI na Digital Ocean

### 1. **Zmienne środowiskowe (ZALECANE dla produkcji)**

#### Digital Ocean App Platform:
1. W panelu Digital Ocean przejdź do swojej aplikacji
2. Wybierz zakładkę **"Settings"**
3. Kliknij **"Environment Variables"**
4. Dodaj nową zmienną:
   - **Key**: `OPENAI_API_KEY`
   - **Value**: `sk-proj-your-actual-api-key-here`
   - **Scope**: Wybierz odpowiedni komponent (web/worker)
5. Zapisz i zredeploy aplikację

#### Digital Ocean Droplet (VPS):
```bash
# Dodaj do ~/.bashrc lub ~/.profile
export OPENAI_API_KEY="sk-proj-your-actual-api-key-here"

# Lub ustaw tymczasowo
export OPENAI_API_KEY="sk-proj-your-actual-api-key-here"
```

### 2. **Streamlit Secrets (dla Streamlit Cloud)**

Utwórz `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-proj-your-actual-api-key-here"
```

### 3. **Plik .env (tylko dla rozwoju)**

**⚠️ UWAGA: NIE zalecane dla produkcji!**

```bash
# .env
OPENAI_API_KEY=sk-proj-your-actual-api-key-here
```

## Dlaczego zmienne środowiskowe są najlepsze?

✅ **Bezpieczeństwo** - klucze nie są w kodzie ani plikach  
✅ **Łatwość zarządzania** - można zmienić bez redeploya kodu  
✅ **Zgodność z DevOps** - standardowe podejście w produkcji  
✅ **Automatyczne ładowanie** - aplikacja automatycznie je pobiera  

## Testowanie konfiguracji

```python
import os
print(f"Klucz API: {'✅ Znaleziony' if os.getenv('OPENAI_API_KEY') else '❌ Brak'}")
```

## Rozwiązywanie problemów

1. **Aplikacja nie znajduje klucza**:
   - Sprawdź czy zmienna środowiskowa jest ustawiona
   - Zrestartuj aplikację po dodaniu zmiennych
   - Sprawdź logi aplikacji

2. **Błąd autoryzacji OpenAI**:
   - Zweryfikuj poprawność klucza API
   - Sprawdź limity i rozliczenia w OpenAI

3. **Local vs Production**:
   - Lokalnie: użyj pliku `.env`
   - Produkcja: ustaw zmienne środowiskowe w panelu Digital Ocean 