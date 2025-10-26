def technische_indicatoren(df, periods, bb_period=20, bb_std_dev=2):
    """
    Print de laatste waarden van SMA, EMA en Bollinger Bands uit een DataFrame.
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

    # === Waarden printen ===
    laatste_datum = df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else df.index[-1]
    print(f"Laatste waarden voor alle indicatoren (datum: {laatste_datum}):")

    for p in periods:
        print(f"- SMA{p}: {df[f'SMA{p}'].iloc[-1]:.2f}")
        print(f"- EMA{p}: {df[f'EMA{p}'].iloc[-1]:.2f}")

    print(f"- BB_Upper: {df['BB_Upper'].iloc[-1]:.2f}")
    print(f"- BB_Middle: {df['BB_Middle'].iloc[-1]:.2f}")
    print(f"- BB_Lower: {df['BB_Lower'].iloc[-1]:.2f}")
