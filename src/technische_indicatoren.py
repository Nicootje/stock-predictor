def technische_indicatoren(df, periods, ticker=None, bb_period=20, bb_std_dev=2):
    """
    Bereken en toon de laatste waarden van prijs, SMA, EMA en Bollinger Bands als nette tabel.
    """
    df = df.copy()

    # === SMA & EMA berekenen ===
    for p in periods:
        df[f'SMA{p}'] = df['Close'].rolling(window=p).mean()
        df[f'EMA{p}'] = df['Close'].ewm(span=p, adjust=False).mean()

    # === Bollinger Bands berekenen ===
    df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
    df['BB_Std'] = df['Close'].rolling(window=bb_period).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * bb_std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * bb_std_dev)

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
