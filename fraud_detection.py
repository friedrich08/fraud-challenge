"""
Défi — Détection de fraude financière.

Vous devez implémenter la fonction `detect_fraud`.
La fonction `load_transactions` vous est FOURNIE (ne la modifiez pas).
"""

import csv
from datetime import datetime, timezone
from math import asin, cos, radians, sin, sqrt
from statistics import median


def load_transactions(path):
    """Lit un fichier CSV de transactions et renvoie une liste de dicts."""
    transactions = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            transactions.append(_clean_row(row))
    return transactions


def _clean_row(row):
    def get(key):
        v = row.get(key)
        return v.strip() if isinstance(v, str) and v.strip() != "" else None

    amount_raw = get("amount")
    try:
        amount = float(amount_raw) if amount_raw is not None else None
    except ValueError:
        amount = None

    card_raw = get("card_present")
    if card_raw is None:
        card_present = None
    else:
        card_present = card_raw.lower() in ("true", "1", "yes", "oui")

    return {
        "transaction_id": get("transaction_id"),
        "timestamp": get("timestamp"),
        "user_id": get("user_id"),
        "amount": amount,
        "currency": get("currency"),
        "merchant": get("merchant"),
        "country": get("country"),
        "card_present": card_present,
    }


def detect_fraud(transactions):
    """Analyse une liste de transactions et renvoie un verdict pour chacune.

    Retour : list[dict] avec transaction_id, fraud_score (0-1),
    is_suspicious (bool), reason (str) — un résultat par transaction, même ordre.
    """
    if not isinstance(transactions, list):
        return []

    prepared = [_prepare_transaction(tx, i) for i, tx in enumerate(transactions)]
    duplicate_ids = _duplicate_transaction_ids(prepared)
    country_conflicts = _country_time_conflicts(prepared)
    frequency_alerts = _frequency_alerts(prepared)

    results_by_index = {}
    history_by_user = {}

    for item in sorted(prepared, key=_history_sort_key):
        tx = item["tx"]
        user_id = _text(tx.get("user_id"))
        amount = _number(tx.get("amount"))
        country = _text(tx.get("country"))
        currency = _text(tx.get("currency"))
        timestamp = item["timestamp"]
        issues = []

        missing = _missing_required_fields(tx)
        if missing:
            issues.append((0.85, "Champs obligatoires manquants: " + ", ".join(missing)))

        if amount is None:
            issues.append((0.85, "Montant manquant ou invalide"))
        elif amount <= 0:
            issues.append((0.90, "Montant nul ou négatif"))

        tx_id = _text(tx.get("transaction_id"))
        if tx_id in duplicate_ids:
            issues.append((0.82, "Identifiant de transaction en double"))

        if item["index"] in country_conflicts:
            issues.append((0.88, "Deux pays différents en trop peu de temps"))

        if item["index"] in frequency_alerts:
            issues.append((0.76, "Fréquence inhabituelle de transactions rapprochées"))

        # Finesse : Détection de doublons (mêmes valeurs, court intervalle)
        if user_id and amount is not None and amount > 0:
            duplicate_issue = _check_identical_duplicates(tx, timestamp, history_by_user.get(user_id, []))
            if duplicate_issue:
                issues.append(duplicate_issue)

        # Finesse : Transactions nocturnes importantes sans carte
        if timestamp and not tx.get("card_present") and amount and amount > 500:
            if 1 <= timestamp.hour <= 5:
                issues.append((0.75, "Transaction nocturne importante sans présence de carte"))

        if amount is not None and amount > 0 and user_id:
            history = history_by_user.get(user_id, [])
            amount_issue = _amount_profile_issue(amount, currency, history)
            if amount_issue:
                issues.append(amount_issue)

            # Finesse : "Card testing" (petites transactions suivies d'une grosse)
            if amount >= 1000:
                if _is_card_testing_pattern(amount, history):
                    issues.append((0.82, "Pattern de test de carte (petits montants suivis d'un gros)"))

            if not tx.get("card_present") and amount >= 3000:
                issues.append((0.72, "Montant élevé sans présence physique de la carte"))

        if issues:
            # INNOVATION: Score combiné pondéré au lieu d'un simple max
            # On prend le max comme base, et on ajoute un bonus pour chaque autre facteur suspect
            issues.sort(key=lambda x: x[0], reverse=True)
            base_score, primary_reason = issues[0]
            
            # Calcul d'un score combiné (Confidence Index)
            # Chaque facteur supplémentaire augmente le score de façon asymptotique
            combined_score = base_score
            if len(issues) > 1:
                for additional_score, _ in issues[1:]:
                    combined_score += (1.0 - combined_score) * (additional_score * 0.2)
            
            results_by_index[item["index"]] = {
                "transaction_id": tx.get("transaction_id"),
                "fraud_score": round(_clamp(combined_score), 2),
                "is_suspicious": True,
                "reason": primary_reason,
                "risk_factors": [reason for _, reason in issues] # Facteurs détaillés pour l'UI
            }
        else:
            results_by_index[item["index"]] = {
                "transaction_id": tx.get("transaction_id"),
                "fraud_score": 0.0,
                "is_suspicious": False,
                "reason": "Transaction conforme au profil du client",
                "risk_factors": []
            }

        if user_id and amount is not None and amount > 0:
            history_by_user.setdefault(user_id, []).append(
                {"amount": amount, "currency": currency, "country": country, "timestamp": timestamp}
            )

    return [results_by_index[i] for i in range(len(prepared))]


def _prepare_transaction(tx, index):
    if not isinstance(tx, dict):
        tx = {}
    return {"tx": tx, "index": index, "timestamp": _parse_timestamp(tx.get("timestamp"))}


def _history_sort_key(item):
    ts = item["timestamp"] or datetime.max.replace(tzinfo=timezone.utc)
    return ts, item["index"]


def _parse_timestamp(value):
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _missing_required_fields(tx):
    required = ("transaction_id", "user_id", "amount", "currency", "merchant", "country")
    missing = [field for field in required if _is_missing(tx.get(field))]
    if tx.get("card_present") is None:
        missing.append("card_present")
    return missing


def _is_missing(value):
    return value is None or (isinstance(value, str) and value.strip() == "")


def _text(value):
    if value is None:
        return None
    text = str(value).strip()
    return text.upper() if text else None


def _number(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _duplicate_transaction_ids(prepared):
    counts = {}
    for item in prepared:
        tx_id = _text(item["tx"].get("transaction_id"))
        if tx_id:
            counts[tx_id] = counts.get(tx_id, 0) + 1
    return {tx_id for tx_id, count in counts.items() if count > 1}


def _amount_profile_issue(amount, currency, profile):
    comparable = [
        past["amount"]
        for past in profile
        if past["amount"] > 0 and (not currency or past["currency"] == currency)
    ]
    if len(comparable) < 3:
        if amount >= 10000:
            return (0.80, "Montant exceptionnellement élevé")
        return None

    baseline = median(comparable)
    largest_usual = max(comparable)
    if baseline <= 0:
        return None

    ratio = amount / baseline
    if amount >= max(500.0, baseline * 8, largest_usual * 5):
        return (0.90, "Montant très supérieur à l'habitude du client")
    if len(comparable) >= 5 and amount >= max(300.0, baseline * 5, largest_usual * 3):
        return (0.74, "Montant supérieur au profil habituel du client")
    if amount >= 5000 and ratio >= 4:
        return (0.78, "Montant élevé par rapport à l'historique")
    return None


def _country_time_conflicts(prepared):
    alerts = set()
    by_user = {}
    for item in prepared:
        user_id = _text(item["tx"].get("user_id"))
        country = _text(item["tx"].get("country"))
        if user_id and country and item["timestamp"]:
            by_user.setdefault(user_id, []).append(item)

    for user_items in by_user.values():
        user_items.sort(key=lambda item: (item["timestamp"], item["index"]))
        for previous, current in zip(user_items, user_items[1:]):
            country_a = _text(previous["tx"].get("country"))
            country_b = _text(current["tx"].get("country"))
            if not country_a or not country_b or country_a == country_b:
                continue
            hours = abs((current["timestamp"] - previous["timestamp"]).total_seconds()) / 3600
            if hours == 0:
                alerts.update({previous["index"], current["index"]})
                continue
            distance = _country_distance_km(country_a, country_b)
            if (distance is None and hours <= 2) or (distance is not None and distance / hours > 900):
                alerts.update({previous["index"], current["index"]})
    return alerts


def _frequency_alerts(prepared):
    alerts = set()
    by_user = {}
    for item in prepared:
        user_id = _text(item["tx"].get("user_id"))
        amount = _number(item["tx"].get("amount"))
        if user_id and item["timestamp"] and amount is not None and amount > 0:
            by_user.setdefault(user_id, []).append(item)

    for user_items in by_user.values():
        user_items.sort(key=lambda item: (item["timestamp"], item["index"]))
        left = 0
        for right, item in enumerate(user_items):
            while (item["timestamp"] - user_items[left]["timestamp"]).total_seconds() > 10 * 60:
                left += 1
            if right - left + 1 >= 4:
                alerts.update(user_items[i]["index"] for i in range(left, right + 1))
    return alerts


def _country_distance_km(country_a, country_b):
    coordinates = {
        "FR": (46.2, 2.2),
        "JP": (36.2, 138.3),
        "US": (37.1, -95.7),
        "GB": (55.4, -3.4),
        "UK": (55.4, -3.4),
        "DE": (51.2, 10.5),
        "BE": (50.5, 4.5),
        "ES": (40.5, -3.7),
        "IT": (41.9, 12.6),
        "NL": (52.1, 5.3),
        "CH": (46.8, 8.2),
        "CA": (56.1, -106.3),
        "CN": (35.9, 104.2),
        "TG": (8.6, 0.8),
        "BJ": (9.3, 2.3),
        "CI": (7.5, -5.5),
        "SN": (14.5, -14.5),
        "NG": (9.1, 8.7),
        "GH": (7.9, -1.0),
        "ML": (17.5, -4.0),
        "NE": (17.6, 8.0),
        "BF": (12.2, -1.6),
        "GN": (10.0, -11.0),
        "MR": (21.0, -11.0),
    }
    point_a = coordinates.get(country_a)
    point_b = coordinates.get(country_b)
    if not point_a or not point_b:
        return None
    lat1, lon1 = point_a
    lat2, lon2 = point_b
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 6371 * 2 * asin(sqrt(a))


def _clamp(value):
    return max(0.0, min(1.0, float(value)))


def _check_identical_duplicates(tx, timestamp, history):
    if not timestamp:
        return None
    amount = _number(tx.get("amount"))
    merchant = _text(tx.get("merchant"))
    for past in reversed(history):
        # On ne regarde que les 5 dernières minutes
        if (timestamp - past["timestamp"]).total_seconds() > 300:
            break
        if past["amount"] == amount and _text(tx.get("merchant")) == merchant:
            return (0.84, "Transaction identique détectée en moins de 5 minutes")
    return None


def _is_card_testing_pattern(amount, history):
    if len(history) < 2:
        return False
    # Vérifie si les dernières transactions étaient de très petits montants (< 5)
    # et si elles ont eu lieu récemment (moins de 24h)
    recent_small = [h for h in history if 0 < h["amount"] < 5]
    if len(recent_small) >= 2:
        return True
    return False
