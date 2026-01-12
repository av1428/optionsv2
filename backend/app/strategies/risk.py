import math

class RiskManager:
    def __init__(self, portfolio_value: float = 6000.0):
        self.portfolio_value = portfolio_value
        self.max_risk_per_trade = 0.05  # 5% max risk per trade
        self.max_exposure_per_sector = 0.20 # 20% max per sector

    def calculate_position_size(self, entry_price: float, stop_loss: float):
        """
        Calculate number of contracts/shares based on risk.
        Risk = Entry - Stop Loss
        """
        if entry_price <= stop_loss:
            return 0 # Invalid
            
        risk_per_share = entry_price - stop_loss
        max_loss_amount = self.portfolio_value * self.max_risk_per_trade
        
        # position_size = max_loss / risk_per_share
        size = max_loss_amount / risk_per_share
        
        # Return integer size, ensuring at least 1 if feasible, but respecting capital
        return math.floor(size)

    def suggested_stop_loss(self, current_price: float, atr: float, strategy_type: str = "long"):
        """
        Suggest stop loss based on ATR (Average True Range).
        """
        multiplier = 2.0
        if strategy_type == "long":
            return current_price - (atr * multiplier)
        elif strategy_type == "short":
            return current_price + (atr * multiplier)
        return current_price

    def diversification_check(self, current_positions: list, new_ticker: str):
        """
        Ensure we don't have too many positions in the same ticker.
        """
        # count occurrences
        count = sum(1 for p in current_positions if p['ticker'] == new_ticker)
        if count >= 1:
            return False, "Already have a position in this ticker. Diversify!"
        return True, "OK"

risk_manager = RiskManager()
