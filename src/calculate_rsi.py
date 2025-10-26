def calculate_rsi(data, period=14):
    """
    Berekent de Relative Strength Index (RSI) voor een gegeven dataset en periode.
    
    Parameters:
    - data: DataFrame met een 'Close'-kolom
    - period: Aantal periodes voor RSI-berekening (standaard 14)
    
    Returns:
    - Series met RSI-waarden
    """
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi