from src.calc_indicators import calc_sma_ema, calc_bollinger_bands, _single_ticker


def technische_indicatoren(df, periods, ticker=None, bb_period=20, bb_std_dev=2):
    """
    Bereken en toon de laatste waarden van prijs, SMA, EMA en Bollinger Bands als nette tabel.
    """
    df = _single_ticker(df, ticker).copy()
    if df.empty:
        raise ValueError("Geen koersdata ontvangen; controleer de downloadmelding.")

    # === SMA & EMA berekenen ===
    df = calc_sma_ema(df, periods)

    # === Bollinger Bands berekenen ===
    df = calc_bollinger_bands(df, bb_period, bb_std_dev)

    # === Laatste rij en datum ===
    laatste_rij = df.iloc[[-1]]
    laatste_datum = laatste_rij.index[-1].strftime('%Y-%m-%d')

    # === Kolommen selecteren ===
    kolommen = (
        ['Close']
        + [f'SMA{p}' for p in periods]
        + [f'EMA{p}' for p in periods]
        + ['BB_Upper', 'BB_Middle', 'BB_Lower']
    )

    tabel = laatste_rij[kolommen].T
    kolomnaam = ticker.upper() if ticker else "Value"
    tabel.columns = [kolomnaam]
    tabel.index.name = f"Indicatoren ({laatste_datum})"

    # === Afronden op 3 decimalen en tonen ===
    tabel = tabel.round(3)

    return tabel
